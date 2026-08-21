#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
MASTER_LOG="logs/v3_unattended_$(date '+%Y%m%d_%H%M%S').log"
STATUS_FILE="logs/v3_unattended_status.txt"
BACKUP_ROOT="${BACKUP_ROOT:-/root/autodl-fs/schemeE_v3_latest}"
SHUTDOWN_ON_SUCCESS="${SHUTDOWN_ON_SUCCESS:-1}"
SHUTDOWN_ON_FAILURE="${SHUTDOWN_ON_FAILURE:-1}"

status() {
  printf 'time=%s\nstatus=%s\nmessage=%s\nlog=%s\n' \
    "$(date '+%F %T')" "$1" "$2" "$MASTER_LOG" > "$STATUS_FILE"
}

poweroff_instance() {
  sync
  if [[ -x /usr/bin/shutdown ]]; then
    /usr/bin/shutdown
  else
    shutdown -h now
  fi
}

backup_results() {
  mkdir -p "$BACKUP_ROOT"
  shopt -s nullglob
  archives=(schemeE_v3_results_*.tar.gz schemeE_v3_results_*.tar.gz.sha256)
  (( ${#archives[@]} )) && cp -f "${archives[@]}" "$BACKUP_ROOT"/
  [[ -s outputs/v3/Round2_Test_Channel.npy ]] && cp -f \
    outputs/v3/Round2_Test_Channel.npy "$BACKUP_ROOT"/
  if [[ -d artifacts/v3 ]]; then
    mkdir -p "$BACKUP_ROOT/artifacts_v3"
    cp -rf artifacts/v3/. "$BACKUP_ROOT/artifacts_v3/"
  fi
  if [[ -d reports/generated ]]; then
    mkdir -p "$BACKUP_ROOT/reports"
    cp -rf reports/generated/. "$BACKUP_ROOT/reports/"
  fi
  if [[ -d logs ]]; then
    mkdir -p "$BACKUP_ROOT/logs"
    cp -rf logs/. "$BACKUP_ROOT/logs/"
  fi
  [[ -s configs/v3_final_selected.json ]] && cp -f \
    configs/v3_final_selected.json "$BACKUP_ROOT"/
  sync
}

on_error() {
  code=$?
  status FAILED "Scheme E-v3 pipeline exited with code $code"
  backup_results || true
  if [[ "$SHUTDOWN_ON_FAILURE" == "1" ]]; then
    poweroff_instance || true
  fi
  exit "$code"
}
trap on_error ERR INT TERM

status RUNNING "Scheme E-v3 Fold attempts and final training are running"
set -o pipefail
bash scripts/run_v3_5090.sh 2>&1 | tee "$MASTER_LOG"
backup_results
status SUCCESS "Scheme E-v3 weights, reports and test NPY were backed up"
trap - ERR INT TERM
if [[ "$SHUTDOWN_ON_SUCCESS" == "1" ]]; then
  poweroff_instance
fi
