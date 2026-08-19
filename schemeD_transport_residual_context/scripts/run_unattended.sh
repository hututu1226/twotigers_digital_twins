#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
MASTER_LOG="logs/unattended_$(date '+%Y%m%d_%H%M%S').log"
STATUS_FILE="logs/unattended_status.txt"
BACKUP_ROOT="${BACKUP_ROOT:-/root/autodl-fs/schemeD_latest}"
SHUTDOWN_ON_SUCCESS="${SHUTDOWN_ON_SUCCESS:-1}"
SHUTDOWN_ON_FAILURE="${SHUTDOWN_ON_FAILURE:-1}"

status() { printf 'time=%s\nstatus=%s\nmessage=%s\nlog=%s\n' "$(date '+%F %T')" "$1" "$2" "$MASTER_LOG" > "$STATUS_FILE"; }
shutdown_instance() { sync; if [[ -x /usr/bin/shutdown ]]; then /usr/bin/shutdown; else shutdown -h now; fi; }
backup() {
  mkdir -p "$BACKUP_ROOT"
  shopt -s nullglob
  archives=(schemeD_results_*.tar.gz schemeD_results_*.tar.gz.sha256)
  (( ${#archives[@]} )) && cp -f "${archives[@]}" "$BACKUP_ROOT"/
  [[ -s outputs/final/Round2_Test_Channel.npy ]] && cp -f outputs/final/Round2_Test_Channel.npy "$BACKUP_ROOT"/
  [[ -d reports/generated ]] && cp -rf reports/generated "$BACKUP_ROOT"/reports
  [[ -s configs/final_selected.json ]] && cp -f configs/final_selected.json "$BACKUP_ROOT"/
  sync
}
on_error() { code=$?; status FAILED "pipeline exited with code $code"; backup || true; [[ "$SHUTDOWN_ON_FAILURE" == 1 ]] && shutdown_instance || true; exit "$code"; }
trap on_error ERR INT TERM
status RUNNING "Scheme D formal pipeline is running"
set -o pipefail
bash scripts/run_all_5090.sh 2>&1 | tee "$MASTER_LOG"
backup
status SUCCESS "weights, report, test NPY and archive were backed up"
trap - ERR INT TERM
[[ "$SHUTDOWN_ON_SUCCESS" == 1 ]] && shutdown_instance
