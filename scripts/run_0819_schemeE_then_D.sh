#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/twotigers_digital_twins}"
BACKUP_ROOT="${BACKUP_ROOT:-/root/autodl-fs/0819_results}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-1}"
cd "$REPO_ROOT"
mkdir -p logs "$BACKUP_ROOT/schemeE" "$BACKUP_ROOT/schemeD"
MASTER_LOG="logs/0819_master_$(date '+%Y%m%d_%H%M%S').log"
STATUS_FILE="logs/0819_master_status.txt"
E_EXIT=not_started
D_EXIT=not_started
RUN_FINISHED=0

write_status() {
  printf 'time=%s\nstage=%s\nschemeE_exit=%s\nschemeD_exit=%s\nlog=%s\n' \
    "$(date '+%F %T')" "$1" "$E_EXIT" "$D_EXIT" "$MASTER_LOG" \
    > "$STATUS_FILE"
}

backup_scheme() {
  local project="$1" destination="$2" label="$3"
  [[ -d "$project" ]] || return 0
  shopt -s nullglob
  local archives=("$project"/${label}_results_*.tar.gz "$project"/${label}_results_*.tar.gz.sha256)
  (( ${#archives[@]} )) && cp -f "${archives[@]}" "$destination"/
  if [[ -s "$project/outputs/final/Round2_Test_Channel.npy" ]]; then
    cp -f "$project/outputs/final/Round2_Test_Channel.npy" "$destination/${label}_Round2_Test_Channel.npy"
  fi
  if [[ -d "$project/reports/generated" ]]; then
    mkdir -p "$destination/reports"
    cp -rf "$project/reports/generated/." "$destination/reports/"
  fi
  [[ -s "$project/configs/final_selected.json" ]] && cp -f "$project/configs/final_selected.json" "$destination"/
}

shutdown_instance() {
  sync
  if [[ -x /usr/bin/shutdown ]]; then /usr/bin/shutdown; else shutdown -h now; fi
}

finalize() {
  local code=$?
  trap - EXIT INT TERM
  backup_scheme schemeE_spectral_gaussian_hybrid "$BACKUP_ROOT/schemeE" schemeE || true
  backup_scheme schemeD_transport_residual_context "$BACKUP_ROOT/schemeD" schemeD || true
  if [[ "$RUN_FINISHED" == "1" ]]; then
    if [[ "$E_EXIT" == "0" && "$D_EXIT" == "0" ]]; then
      write_status complete
    else
      write_status complete_with_failures
    fi
  else
    write_status interrupted
  fi
  sync
  if [[ "$SHUTDOWN_WHEN_DONE" == "1" ]]; then
    shutdown_instance || true
  fi
  exit "$code"
}

trap finalize EXIT
trap 'exit 130' INT TERM

exec > >(tee -a "$MASTER_LOG") 2>&1
write_status schemeE_running
echo "[$(date '+%F %T')] START Scheme E"
(cd schemeE_spectral_gaussian_hybrid && bash scripts/run_all_5090.sh)
E_EXIT=$?
backup_scheme schemeE_spectral_gaussian_hybrid "$BACKUP_ROOT/schemeE" schemeE || true
echo "[$(date '+%F %T')] END Scheme E exit=$E_EXIT"

write_status schemeD_running
echo "[$(date '+%F %T')] START Scheme D (runs even when Scheme E failed)"
(cd schemeD_transport_residual_context && bash scripts/run_all_5090.sh)
D_EXIT=$?
backup_scheme schemeD_transport_residual_context "$BACKUP_ROOT/schemeD" schemeD || true
echo "[$(date '+%F %T')] END Scheme D exit=$D_EXIT"

RUN_FINISHED=1
if [[ "$E_EXIT" == "0" && "$D_EXIT" == "0" ]]; then
  exit 0
fi
exit 1
