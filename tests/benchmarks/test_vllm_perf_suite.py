# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
from argparse import Namespace

from benchmarks.vllm_perf_suite import (
    SweepPoint,
    build_bench_command,
    build_gsm8k_prompts,
    build_result_filename,
    create_accuracy_payload,
    extract_answer_value,
    load_result_rows,
    parse_accuracy_response,
    parse_max_concurrencies,
    parse_request_rates,
)


def test_parse_request_rates_supports_inf() -> None:
    assert parse_request_rates("1,2.5,inf") == [1.0, 2.5, float("inf")]


def test_parse_max_concurrencies_supports_none() -> None:
    assert parse_max_concurrencies("1,4,none") == [1, 4, None]


def test_build_bench_command_includes_goodput_and_random_args() -> None:
    args = Namespace(
        vllm_bin=".venv/bin/vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        host="127.0.0.1",
        port=8000,
        endpoint="/v1/completions",
        backend="vllm",
        dataset_name="random",
        dataset_path=None,
        num_prompts=32,
        input_len=512,
        output_len=64,
        label="suite-a",
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="50,99",
        result_dir="log/results",
        served_model_name="qwen-serving-name",
        disable_tqdm=True,
        goodput=["ttft:800", "e2el:2500"],
        bench_arg=["--temperature", "0"],
    )

    command = build_bench_command(
        args,
        SweepPoint(request_rate=4.0, max_concurrency=2),
        build_result_filename("suite-a", SweepPoint(4.0, 2)),
    )

    assert "--goodput" in command
    assert "ttft:800" in command
    assert "e2el:2500" in command
    assert "--random-input-len" in command
    assert "--random-output-len" in command
    assert "--max-concurrency" in command
    assert "--temperature" in command


def test_extract_answer_value_uses_last_number() -> None:
    assert extract_answer_value("steps... answer is 1,234 therefore 56") == 56
    assert extract_answer_value("no digits") < 0


def test_build_gsm8k_prompts_and_labels() -> None:
    train_rows = [
        {"question": "1+1?", "answer": "2"},
        {"question": "2+2?", "answer": "4"},
    ]
    test_rows = [
        {"question": "3+3?", "answer": "6"},
        {"question": "4+4?", "answer": "8"},
    ]

    prompts, labels = build_gsm8k_prompts(train_rows, test_rows, 2, 2)

    assert len(prompts) == 2
    assert labels == [6, 8]
    assert "Question: 1+1?" in prompts[0]
    assert prompts[0].endswith("Question: 3+3?\nAnswer:")


def test_create_accuracy_payload_and_parse_chat_response() -> None:
    endpoint, payload = create_accuracy_payload(
        "openai-chat",
        "model-a",
        "prompt text",
        128,
        0.0,
    )

    assert endpoint == "/v1/chat/completions"
    assert payload["model"] == "model-a"
    assert payload["messages"][0]["content"] == "prompt text"

    text, tokens = parse_accuracy_response(
        "openai-chat",
        {
            "choices": [{"message": {"content": "final answer 42"}}],
            "usage": {"completion_tokens": 17},
        },
    )

    assert text == "final answer 42"
    assert tokens == 17


def test_load_result_rows_extracts_summary_fields(tmp_path) -> None:
    payload = {
        "label": "suite-a",
        "backend": "vllm",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "request_rate": 8.0,
        "max_concurrency": 4,
        "completed": 32,
        "failed": 0,
        "request_throughput": 7.8,
        "request_goodput": 7.2,
        "median_ttft_ms": 88.4,
        "p99_ttft_ms": 120.5,
        "median_tpot_ms": 19.2,
        "p99_tpot_ms": 23.1,
        "p99_e2el_ms": 910.3,
    }
    result_file = tmp_path / "suite-a-rps-8-conc-4.json"
    result_file.write_text(json.dumps(payload), encoding="utf-8")

    rows = load_result_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["file"] == result_file.name
    assert rows[0]["request_goodput"] == 7.2
    assert rows[0]["p99_ttft_ms"] == 120.5
