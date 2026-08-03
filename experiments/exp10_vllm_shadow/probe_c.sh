#!/usr/bin/env bash
# 等 vLLM server ready + 发一次 B=16 test request
# 使用: ./probe_c.sh [PORT]  (默认 8000)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8000}"

LATEST=$(ls -td run_*/ 2>/dev/null | head -1)
if [[ -z "$LATEST" ]]; then
  echo "no run_*/ dir found" >&2
  exit 1
fi
LOG="$LATEST/server.log"

echo "=== waiting for server ready (Uvicorn banner) ==="
for i in $(seq 1 60); do  # 最多 60 × 10s = 10 分钟
  if grep -q "Uvicorn running" "$LOG" 2>/dev/null; then
    echo "READY (took ${i}0s)"
    break
  fi
  sleep 10
  echo -n "."
done

if ! grep -q "Uvicorn running" "$LOG"; then
  echo ""
  echo "TIMEOUT — check $LOG for errors"
  tail -20 "$LOG"
  exit 1
fi

echo ""
echo "=== sending B=16 test request ==="

MODEL_NAME=$(grep -oP 'served-model-name[^\s]*|--model \K\S+' "$LOG" 2>/dev/null | head -1)
[[ -z "$MODEL_NAME" ]] && MODEL_NAME="/models/DeepSeek-V3.2"

# 用 completions API,一次发 16 条 prompt(每条独立 request → B=16 batch)
BODY=$(python3 -c "
import json
prompts = ['你好,请介绍一下你自己'] * 16
body = {
    'model': '$MODEL_NAME',
    'prompt': prompts,
    'max_tokens': 4,
    'temperature': 0.0,
}
print(json.dumps(body))
")

curl -sN "localhost:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  --max-time 60 \
  -d "$BODY" \
  | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(f\"got {len(d.get('choices',[]))} responses\")
for c in d.get('choices', [])[:3]:
    print(f\"  [{c.get('index','?')}] {c.get('text','')[:60]!r}\")
"

echo ""
echo "=== dump files landed ==="
ls -la "$LATEST"/call_*.pt 2>/dev/null || echo "no call_*.pt yet"
