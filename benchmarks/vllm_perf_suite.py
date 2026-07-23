#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for running and summarizing vLLM serving benchmarks.

This script is a thin wrapper around ``vllm bench serve``. It exists to make
common TTFT, TPOT, and goodput sweeps repeatable, and to emit a compact CSV
summary that is easy to compare across runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import regex as re

DEFAULT_GOODPUT_SLOS = ["ttft:1000", "tpot:200", "e2el:3000"]
DEFAULT_PERCENTILES = "50,90,95,99"
DEFAULT_PERCENTILE_METRICS = "ttft,tpot,itl,e2el"
GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/train.jsonl"
)
GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
INVALID_ANSWER = -9999999
SUMMARY_FIELDS = [
    "file",
    "label",
    "backend",
    "model_id",
    "request_rate",
    "max_concurrency",
    "completed",
    "failed",
    "request_throughput",
    "request_goodput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
    "output_throughput",
    "total_token_throughput",
]
EVALUATION_FIELDS = [
    "dataset",
    "model_id",
    "accuracy",
    "invalid_rate",
    "num_questions",
    "num_shots",
    "accuracy_latency",
    "questions_per_second",
    "accuracy_output_tokens",
    "accuracy_tokens_per_second",
    "request_rate",
    "max_concurrency",
    "request_throughput",
    "request_goodput",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
    "output_throughput",
    "total_token_throughput",
    "performance_result_file",
]
LOCAL_GSM8K_BENCHMARK_FILENAME = "gsm8k_benchmark_prompts.jsonl"


@dataclass(frozen=True)
class SweepPoint:
    request_rate: float
    max_concurrency: int | None


def http_get_json(url: str, timeout_seconds: float = 10) -> dict[str, object]:
    with urlopen(url, timeout=timeout_seconds) as response:
        return json.load(response)


def http_post_json(
    url: str,
    payload: dict[str, object],
    timeout_seconds: float = 600,
) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.load(response)


def detect_model_id(base_url: str, timeout_seconds: float = 10) -> str | None:
    model_info = detect_model_info(base_url, timeout_seconds)
    if model_info is None:
        return None
    return model_info.get("id")


def detect_model_info(
    base_url: str,
    timeout_seconds: float = 10,
) -> dict[str, str] | None:
    try:
        payload = http_get_json(
            f"{base_url.rstrip('/')}/v1/models",
            timeout_seconds=timeout_seconds,
        )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    model_info = data[0]
    if not isinstance(model_info, dict):
        return None
    model_id = model_info.get("id")
    root = model_info.get("root")
    result: dict[str, str] = {}
    if model_id:
        result["id"] = str(model_id)
    if root:
        result["root"] = str(root)
    return result or None


def parse_request_rates(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value == "inf":
            values.append(float("inf"))
            continue
        rate = float(value)
        if rate <= 0:
            raise ValueError("request rate must be positive")
        values.append(rate)
    if not values:
        raise ValueError("at least one request rate is required")
    return values


def parse_max_concurrencies(raw: str | None) -> list[int | None]:
    if raw is None or not raw.strip():
        return [None]

    values: list[int | None] = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value == "none":
            values.append(None)
            continue
        concurrency = int(value)
        if concurrency <= 0:
            raise ValueError("max concurrency must be positive")
        values.append(concurrency)
    if not values:
        raise ValueError("at least one max concurrency value is required")
    return values


def wait_for_server(base_url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    health_url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except URLError:
            pass
        time.sleep(1)
    raise TimeoutError(f"server did not become ready: {health_url}")


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    with urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())
    return destination


def ensure_gsm8k_dataset(dataset_dir: Path) -> tuple[Path, Path]:
    train_path = download_file(GSM8K_TRAIN_URL, dataset_dir / "train.jsonl")
    test_path = download_file(GSM8K_TEST_URL, dataset_dir / "test.jsonl")
    return train_path, test_path


def get_local_gsm8k_dataset(dataset_dir: Path) -> tuple[Path, Path]:
    train_path = dataset_dir / "train.jsonl"
    test_path = dataset_dir / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "local GSM8K dataset not found; download train.jsonl and test.jsonl "
            f"into {dataset_dir} first"
        )
    return train_path, test_path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def extract_answer_value(answer_text: str) -> int:
    normalized = answer_text.replace(",", "")
    matches = re.findall(r"\d+", normalized)
    if not matches:
        return INVALID_ANSWER
    try:
        return int(matches[-1])
    except ValueError:
        return INVALID_ANSWER


def build_gsm8k_prompts(
    train_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    num_questions: int,
    num_shots: int,
) -> tuple[list[str], list[int]]:
    few_shot_examples = ""
    for index in range(num_shots):
        few_shot_examples += (
            f"Question: {train_rows[index]['question']}\n"
            f"Answer: {train_rows[index]['answer']}\n\n"
        )

    prompts: list[str] = []
    labels: list[int] = []
    for row in test_rows[:num_questions]:
        prompts.append(few_shot_examples + f"Question: {row['question']}\nAnswer:")
        labels.append(extract_answer_value(str(row["answer"])))
    return prompts, labels


def materialize_local_gsm8k_benchmark_dataset(
    dataset_dir: Path,
    test_rows: list[dict[str, object]],
    num_prompts: int,
    output_tokens: int,
) -> Path:
    dataset_path = dataset_dir / LOCAL_GSM8K_BENCHMARK_FILENAME
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("w", encoding="utf-8") as file:
        for row in test_rows[:num_prompts]:
            payload = {
                "prompt": str(row["question"]),
                "output_tokens": output_tokens,
            }
            file.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return dataset_path


def format_rate_for_filename(rate: float) -> str:
    if math.isinf(rate):
        return "inf"
    if float(rate).is_integer():
        return str(int(rate))
    return str(rate).replace(".", "p")


def build_result_filename(label: str, point: SweepPoint) -> str:
    rate = format_rate_for_filename(point.request_rate)
    concurrency = "none" if point.max_concurrency is None else point.max_concurrency
    return f"{label}-rps-{rate}-conc-{concurrency}.json"


def build_bench_command(
    args: argparse.Namespace,
    point: SweepPoint,
    result_filename: str,
) -> list[str]:
    command = [
        args.vllm_bin,
        "bench",
        "serve",
        "--backend",
        args.backend,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--endpoint",
        args.endpoint,
        "--dataset-name",
        args.dataset_name,
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        "inf" if math.isinf(point.request_rate) else str(point.request_rate),
        "--percentile-metrics",
        args.percentile_metrics,
        "--metric-percentiles",
        args.metric_percentiles,
        "--save-result",
        "--result-dir",
        str(args.result_dir),
        "--result-filename",
        result_filename,
        "--metadata",
        f"suite_label={args.label}",
    ]

    if args.model:
        command.extend(["--model", args.model])

    if getattr(args, "tokenizer", None):
        command.extend(["--tokenizer", args.tokenizer])

    if args.served_model_name:
        command.extend(["--served-model-name", args.served_model_name])

    if getattr(args, "skip_tokenizer_init", False):
        command.append("--skip-tokenizer-init")

    if point.max_concurrency is not None:
        command.extend(["--max-concurrency", str(point.max_concurrency)])

    if args.dataset_path:
        command.extend(["--dataset-path", args.dataset_path])

    if args.dataset_name == "random":
        command.extend(
            [
                "--random-input-len",
                str(args.input_len),
                "--random-output-len",
                str(args.output_len),
                "--ignore-eos",
            ]
        )
    elif args.input_len is not None:
        command.extend(["--input-len", str(args.input_len)])
        if args.output_len is not None:
            command.extend(["--output-len", str(args.output_len)])

    if args.disable_tqdm:
        command.append("--disable-tqdm")

    if args.goodput:
        command.append("--goodput")
        command.extend(args.goodput)

    if args.bench_arg:
        command.extend(args.bench_arg)

    return command


def create_accuracy_payload(
    backend: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, object]]:
    stop = ["Question", "Assistant:", "<|separator|>"]
    if backend == "openai-chat":
        endpoint = "/v1/chat/completions"
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
        }
        return endpoint, payload

    endpoint = "/v1/completions"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    return endpoint, payload


def parse_accuracy_response(
    backend: str, payload: dict[str, object]
) -> tuple[str, int]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", 0

    choice = choices[0]
    if not isinstance(choice, dict):
        return "", 0

    if backend == "openai-chat":
        message = choice.get("message")
        if isinstance(message, dict):
            text = message.get("content", "")
            if isinstance(text, list):
                text = "".join(
                    part.get("text", "") for part in text if isinstance(part, dict)
                )
            text = str(text)
        else:
            text = str(choice.get("text", ""))
    else:
        text = str(choice.get("text", ""))

    usage = payload.get("usage")
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens", 0)
    else:
        completion_tokens = 0
    return text, int(completion_tokens or 0)


def evaluate_accuracy(args: argparse.Namespace, model_id: str) -> dict[str, object]:
    train_path, test_path = get_local_gsm8k_dataset(args.dataset_dir)
    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    prompts, labels = build_gsm8k_prompts(
        train_rows,
        test_rows,
        num_questions=args.accuracy_num_questions,
        num_shots=args.accuracy_num_shots,
    )
    base_url = f"http://{args.host}:{args.port}".rstrip("/")

    def request_one(prompt: str) -> tuple[str, int]:
        endpoint, payload = create_accuracy_payload(
            args.backend,
            model_id,
            prompt,
            args.accuracy_max_tokens,
            args.temperature,
        )
        response = http_post_json(
            f"{base_url}{endpoint}",
            payload,
            timeout_seconds=args.request_timeout_seconds,
        )
        return parse_accuracy_response(args.backend, response)

    started_at = time.perf_counter()
    outputs: list[tuple[str, int]] = [("", 0)] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.accuracy_concurrency
    ) as executor:
        futures = {
            executor.submit(request_one, prompt): index
            for index, prompt in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                outputs[index] = future.result()
            except Exception:
                outputs[index] = ("", 0)
    latency = time.perf_counter() - started_at

    predictions = [extract_answer_value(text) for text, _ in outputs]
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    invalid = sum(int(pred == INVALID_ANSWER) for pred in predictions)
    total_output_tokens = sum(tokens for _, tokens in outputs)
    num_questions = len(labels)

    return {
        "dataset": "gsm8k",
        "model_id": model_id,
        "accuracy": correct / num_questions if num_questions else 0.0,
        "invalid_rate": invalid / num_questions if num_questions else 0.0,
        "num_questions": num_questions,
        "num_shots": args.accuracy_num_shots,
        "accuracy_latency": latency,
        "questions_per_second": num_questions / latency if latency > 0 else 0.0,
        "accuracy_output_tokens": total_output_tokens,
        "accuracy_tokens_per_second": (
            total_output_tokens / latency if latency > 0 else 0.0
        ),
    }


def extract_perf_metrics(result_path: Path) -> dict[str, object]:
    with result_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return {
        "request_rate": payload.get("request_rate"),
        "max_concurrency": payload.get("max_concurrency"),
        "request_throughput": payload.get("request_throughput"),
        "request_goodput": payload.get("request_goodput"),
        "median_ttft_ms": payload.get("median_ttft_ms"),
        "p99_ttft_ms": payload.get("p99_ttft_ms"),
        "median_tpot_ms": payload.get("median_tpot_ms"),
        "p99_tpot_ms": payload.get("p99_tpot_ms"),
        "median_e2el_ms": payload.get("median_e2el_ms"),
        "p99_e2el_ms": payload.get("p99_e2el_ms"),
        "output_throughput": payload.get("output_throughput"),
        "total_token_throughput": payload.get("total_token_throughput"),
        "performance_result_file": result_path.name,
    }


def write_evaluation_report(result_dir: Path, report: dict[str, object]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "served_eval_gsm8k.json"
    csv_path = result_dir / "served_eval_gsm8k.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=True, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerow({field: report.get(field) for field in EVALUATION_FIELDS})


def evaluate_gsm8k_command(args: argparse.Namespace) -> int:
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{args.host}:{args.port}"
    wait_for_server(base_url, args.ready_timeout_seconds)
    served_model_info = detect_model_info(base_url, args.ready_timeout_seconds)
    model_id = args.model or (served_model_info or {}).get("id")
    if not model_id:
        raise RuntimeError("unable to detect model id from /v1/models; pass --model")

    train_path, test_path = ensure_gsm8k_dataset(args.dataset_dir)
    test_rows = read_jsonl(test_path)
    accuracy_report = evaluate_accuracy(args, model_id)
    benchmark_dataset_path = materialize_local_gsm8k_benchmark_dataset(
        args.dataset_dir,
        test_rows,
        args.perf_num_prompts,
        args.accuracy_max_tokens,
    )

    perf_label = args.label
    perf_point = SweepPoint(
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
    )
    result_filename = build_result_filename(perf_label, perf_point)
    perf_args = argparse.Namespace(**vars(args))
    perf_args.model = model_id
    perf_args.dataset_name = "custom"
    perf_args.dataset_path = str(benchmark_dataset_path)
    perf_args.num_prompts = args.perf_num_prompts
    perf_args.request_rates = None
    perf_args.max_concurrencies = None
    perf_args.input_len = None
    perf_args.output_len = args.accuracy_max_tokens
    tokenizer_root = (served_model_info or {}).get("root")
    if tokenizer_root and Path(tokenizer_root).exists():
        perf_args.tokenizer = tokenizer_root
        perf_args.skip_tokenizer_init = False
    else:
        perf_args.tokenizer = None
        perf_args.skip_tokenizer_init = True

    command = build_bench_command(perf_args, perf_point, result_filename)
    log_path = args.log_dir / result_filename.replace(".json", ".log")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"performance benchmark failed, see log: {log_path}")

    perf_report = extract_perf_metrics(args.result_dir / result_filename)
    report = {**accuracy_report, **perf_report}
    write_evaluation_report(args.result_dir, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


def run_sweep(args: argparse.Namespace) -> int:
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{args.host}:{args.port}"
    wait_for_server(base_url, args.ready_timeout_seconds)

    request_rates = parse_request_rates(args.request_rates)
    max_concurrencies = parse_max_concurrencies(args.max_concurrencies)
    points = [
        SweepPoint(request_rate=request_rate, max_concurrency=max_concurrency)
        for request_rate in request_rates
        for max_concurrency in max_concurrencies
    ]

    failures = 0
    for index, point in enumerate(points, start=1):
        result_filename = build_result_filename(args.label, point)
        command = build_bench_command(args, point, result_filename)
        log_path = args.log_dir / result_filename.replace(".json", ".log")

        print(
            "[run {}/{}] request_rate={} max_concurrency={} -> {}".format(
                index,
                len(points),
                "inf" if math.isinf(point.request_rate) else point.request_rate,
                point.max_concurrency,
                result_filename,
            )
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            failures += 1
            print(f"benchmark failed, see log: {log_path}", file=sys.stderr)

    summary_path = args.result_dir / "summary.csv"
    summarize_results(args.result_dir, summary_path)
    print(f"summary written to {summary_path}")
    return 1 if failures else 0


def coerce_sort_value(value: object) -> tuple[int, float | str]:
    if value is None:
        return (2, "")
    if value == "inf":
        return (1, float("inf"))
    if isinstance(value, (int, float)):
        return (0, float(value))
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def load_result_rows(result_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(result_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        row = {field: payload.get(field) for field in SUMMARY_FIELDS}
        row["file"] = path.name
        rows.append(row)

    rows.sort(
        key=lambda row: (
            coerce_sort_value(row.get("request_rate")),
            coerce_sort_value(row.get("max_concurrency")),
            str(row.get("file")),
        )
    )
    return rows


def summarize_results(
    result_dir: Path, summary_path: Path | None = None
) -> list[dict[str, object]]:
    rows = load_result_rows(result_dir)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def print_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("no result json files found")
        return

    display_fields = [
        "file",
        "request_rate",
        "max_concurrency",
        "request_throughput",
        "request_goodput",
        "median_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "p99_e2el_ms",
    ]
    widths = {field: len(field) for field in display_fields}
    rendered_rows: list[dict[str, str]] = []
    for row in rows:
        rendered: dict[str, str] = {}
        for field in display_fields:
            value = row.get(field)
            if isinstance(value, float) and not math.isinf(value):
                text = f"{value:.2f}"
            else:
                text = "" if value is None else str(value)
            widths[field] = max(widths[field], len(text))
            rendered[field] = text
        rendered_rows.append(rendered)

    header = " | ".join(field.ljust(widths[field]) for field in display_fields)
    divider = "-+-".join("-" * widths[field] for field in display_fields)
    print(header)
    print(divider)
    for row in rendered_rows:
        print(" | ".join(row[field].ljust(widths[field]) for field in display_fields))


def summarize_command(args: argparse.Namespace) -> int:
    rows = summarize_results(args.result_dir, args.summary_csv)
    print_summary(rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and summarize repeatable vLLM serving benchmarks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a benchmark sweep")
    run_parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--served-model-name", default=None)
    run_parser.add_argument("--backend", default="vllm")
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, default=8000)
    run_parser.add_argument("--endpoint", default="/v1/completions")
    run_parser.add_argument("--label", default="vllm-perf-suite")
    run_parser.add_argument("--dataset-name", default="random")
    run_parser.add_argument("--dataset-path", default=None)
    run_parser.add_argument("--input-len", type=int, default=1024)
    run_parser.add_argument("--output-len", type=int, default=128)
    run_parser.add_argument("--num-prompts", type=int, default=200)
    run_parser.add_argument("--request-rates", default="1,2,4,8,16")
    run_parser.add_argument("--max-concurrencies", default="1,2,4,8")
    run_parser.add_argument(
        "--goodput",
        nargs="*",
        default=DEFAULT_GOODPUT_SLOS,
        help="SLOs forwarded to --goodput, for example ttft:800 tpot:120",
    )
    run_parser.add_argument(
        "--percentile-metrics",
        default=DEFAULT_PERCENTILE_METRICS,
    )
    run_parser.add_argument(
        "--metric-percentiles",
        default=DEFAULT_PERCENTILES,
    )
    run_parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("log/vllm_bench_results"),
    )
    run_parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log/vllm_bench_logs"),
    )
    run_parser.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=180,
    )
    run_parser.add_argument("--disable-tqdm", action="store_true")
    run_parser.add_argument(
        "--bench-arg",
        action="append",
        default=[],
        help="Raw extra argument forwarded to vllm bench serve.",
    )
    run_parser.set_defaults(handler=run_sweep)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="summarize result json files",
    )
    summarize_parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("log/vllm_bench_results"),
    )
    summarize_parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("log/vllm_bench_results/summary.csv"),
    )
    summarize_parser.set_defaults(handler=summarize_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate-gsm8k",
        help=(
            "download GSM8K, run accuracy evaluation, then run a performance benchmark"
        ),
    )
    evaluate_parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    evaluate_parser.add_argument("--model", default=None)
    evaluate_parser.add_argument("--served-model-name", default=None)
    evaluate_parser.add_argument("--backend", default="vllm")
    evaluate_parser.add_argument("--host", default="127.0.0.1")
    evaluate_parser.add_argument("--port", type=int, default=8000)
    evaluate_parser.add_argument("--endpoint", default="/v1/completions")
    evaluate_parser.add_argument("--label", default="served-gsm8k-eval")
    evaluate_parser.add_argument(
        "--request-rate",
        type=float,
        default=4.0,
    )
    evaluate_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
    )
    evaluate_parser.add_argument(
        "--perf-num-prompts",
        type=int,
        default=200,
    )
    evaluate_parser.add_argument(
        "--accuracy-num-questions",
        type=int,
        default=200,
    )
    evaluate_parser.add_argument(
        "--accuracy-num-shots",
        type=int,
        default=5,
    )
    evaluate_parser.add_argument(
        "--accuracy-max-tokens",
        type=int,
        default=256,
    )
    evaluate_parser.add_argument(
        "--accuracy-concurrency",
        type=int,
        default=8,
    )
    evaluate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    evaluate_parser.add_argument(
        "--goodput",
        nargs="*",
        default=DEFAULT_GOODPUT_SLOS,
    )
    evaluate_parser.add_argument(
        "--percentile-metrics",
        default=DEFAULT_PERCENTILE_METRICS,
    )
    evaluate_parser.add_argument(
        "--metric-percentiles",
        default=DEFAULT_PERCENTILES,
    )
    evaluate_parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("log/vllm_bench_results"),
    )
    evaluate_parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log/vllm_bench_logs"),
    )
    evaluate_parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("log/datasets/gsm8k"),
    )
    evaluate_parser.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=180,
    )
    evaluate_parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=600,
    )
    evaluate_parser.add_argument("--disable-tqdm", action="store_true")
    evaluate_parser.add_argument(
        "--bench-arg",
        action="append",
        default=[],
    )
    evaluate_parser.set_defaults(handler=evaluate_gsm8k_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
