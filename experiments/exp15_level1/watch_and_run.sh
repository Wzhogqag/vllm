#!/usr/bin/env bash
# exp15 自动看守:等到 8 张卡全空闲(每张 used<500MB)→ 跑 run_e2e.sh + probe_e2e.sh。
# 用户授权:"只要有空闲8卡,你自主迭代"。所以不抢别人 GPU0 上的活,等它退了再上。
set -u
cd "$(dirname "$0")"
STAMP=runs/watch_result.txt
: > "$STAMP"

free_gpus() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '{if($1+0<500) f++} END{print f+0}'
}

echo "[watch] $(date '+%F %T') start; waiting for 8 free GPUs ..." | tee -a "$STAMP"
# 最多等 6 小时(360 * 60s);够我不在时等别人的活跑完。
for i in $(seq 1 360); do
  n=$(free_gpus)
  if [ "$n" -ge 8 ]; then
    echo "[watch] $(date '+%F %T') 8 GPUs free (iter $i) — launching e2e" | tee -a "$STAMP"
    # 清理任何我的残留
    for pf in runs/e2e_serve.pid runs/e2e_server.pid runs/clean.pid; do
      [ -f "$pf" ] && kill -9 "$(cat "$pf")" 2>/dev/null || true
    done
    pkill -9 -f "launch_clean.py" 2>/dev/null || true
    pkill -9 -f "serve_remote.py" 2>/dev/null || true
    sleep 3
    bash run_e2e.sh >> "$STAMP" 2>&1
    bash probe_e2e.sh >> "$STAMP" 2>&1
    rc=$?
    echo "[watch] $(date '+%F %T') probe exit=$rc" | tee -a "$STAMP"
    # 跑完把服务停掉,别占着 8 卡不放(别人可能要用)
    pkill -9 -f "launch_clean.py" 2>/dev/null || true
    pkill -9 -f "serve_remote.py" 2>/dev/null || true
    echo "[watch] done, servers stopped" | tee -a "$STAMP"
    exit $rc
  fi
  sleep 60
done
echo "[watch] $(date '+%F %T') timed out waiting for 8 GPUs (6h)" | tee -a "$STAMP"
exit 3
