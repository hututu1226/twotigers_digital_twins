#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

if [[ ! -x /usr/bin/shutdown ]]; then
  printf '/usr/bin/shutdown is unavailable; refusing to launch unattended training.\n' >&2
  exit 1
fi

python scripts/verify_completion.py --stage capacity
rm -f NO_AUTO_SHUTDOWN

nohup env \
  PROJECT_DIR="$PROJECT_DIR" \
  PIPELINE_MODE=ae \
  CONFIRM_AUTODL_SHUTDOWN=YES \
  SHUTDOWN_ON_SUCCESS=1 \
  SHUTDOWN_ON_FAILURE=1 \
  bash scripts/run_overnight_pipeline.sh \
  > logs/overnight_launcher.log 2>&1 &

pipeline_pid=$!
printf '%s\n' "$pipeline_pid" > logs/overnight.pid
sleep 3

if ! kill -0 "$pipeline_pid" 2>/dev/null; then
  printf 'AE automation exited during startup. Recent launcher log:\n' >&2
  tail -n 50 logs/overnight_launcher.log >&2 || true
  exit 1
fi

printf 'AE automation started successfully. PID=%s\n' "$pipeline_pid"
printf 'Status: cat logs/overnight_status.txt\n'
printf 'Log:    tail -n 50 -f logs/overnight_pipeline.log\n'
printf 'Cancel automatic shutdown only: touch NO_AUTO_SHUTDOWN\n'

