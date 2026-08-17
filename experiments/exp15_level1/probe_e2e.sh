#!/usr/bin/env bash
# exp15 端到端探针:等 vLLM ready → 打固定 prompt(temp=0)→ 抓输出 → 和 baseline 比对。
# baseline(remote OFF)completion 记录在 runs/clean_resp.txt:"\n\n张量并行是一种并行"。
set -u
cd "$(dirname "$0")"
LOG=runs/e2e_server.log
PROMPT='请用一句话解释什么是张量并行。'
BASELINE=$'\n\n张量并行是一种并行'

echo "[probe] waiting for vLLM readiness ..."
for i in $(seq 1 180); do
  if grep -qE "Application startup complete|Uvicorn running on" "$LOG" 2>/dev/null; then
    echo "[probe] server ready after ~${i}0s"; break
  fi
  if grep -qE "Traceback|Error|error" "$LOG" 2>/dev/null | head -1; then
    : # keep waiting; errors may be benign warnings
  fi
  sleep 10
done

if ! grep -qE "Application startup complete|Uvicorn running on" "$LOG" 2>/dev/null; then
  echo "[probe] FAILED: server not ready — tail of log:"; tail -25 "$LOG"; exit 2
fi

echo "[probe] sending request (temp=0, max_tokens=8) ..."
RESP=$(curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"/models/DeepSeek-V3.2\",\"prompt\":\"$PROMPT\",\"max_tokens\":8,\"temperature\":0}")
echo "$RESP" > runs/e2e_resp.txt
echo "[probe] raw response saved to runs/e2e_resp.txt"

TEXT=$(echo "$RESP" | ../../.venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
echo "[probe] remote-ON  completion: $(printf '%q' "$TEXT")"
echo "[probe] baseline   completion: $(printf '%q' "$BASELINE")"
if [ "$TEXT" = "$BASELINE" ]; then
  echo "[probe] ✅ TOKEN-IDENTICAL — remote indexer produces the same output as baseline"
  exit 0
else
  echo "[probe] ❌ MISMATCH — see runs/e2e_resp.txt and runs/e2e_serve.log for frame diag"
  exit 1
fi
