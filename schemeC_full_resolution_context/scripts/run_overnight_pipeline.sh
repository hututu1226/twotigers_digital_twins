#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SHUTDOWN_ON_SUCCESS="${SHUTDOWN_ON_SUCCESS:-1}"
SHUTDOWN_ON_FAILURE="${SHUTDOWN_ON_FAILURE:-1}"
CONFIRM_AUTODL_SHUTDOWN="${CONFIRM_AUTODL_SHUTDOWN:-NO}"
BACKUP_DIR="${BACKUP_DIR:-/root/autodl-fs}"
PIPELINE_MODE="${PIPELINE_MODE:-fold0}"
MASTER_LOG="$PROJECT_DIR/logs/overnight_pipeline.log"
STATUS_FILE="$PROJECT_DIR/logs/overnight_status.txt"
NO_SHUTDOWN_FILE="$PROJECT_DIR/NO_AUTO_SHUTDOWN"
CURRENT_STAGE="initialization"

cd "$PROJECT_DIR"
mkdir -p logs

if [[ "$PIPELINE_MODE" != "ae" && "$PIPELINE_MODE" != "fold0" && "$PIPELINE_MODE" != "final" && "$PIPELINE_MODE" != "all" ]]; then
    printf 'PIPELINE_MODE must be ae, fold0, final, or all; got %s\n' "$PIPELINE_MODE" >&2
    exit 2
fi

exec 9>logs/overnight_pipeline.lock
if ! flock -n 9; then
    printf 'Another overnight Scheme C pipeline is already running.\n' >&2
    exit 1
fi

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"
}

write_status() {
    local state="$1"
    local message="$2"
    printf 'state=%s\nstage=%s\ntime=%s\nmessage=%s\n' \
        "$state" "$CURRENT_STAGE" "$(date --iso-8601=seconds)" "$message" \
        > "$STATUS_FILE"
}

shutdown_instance() {
    local reason="$1"
    if [[ -f "$NO_SHUTDOWN_FILE" ]]; then
        log "Automatic shutdown cancelled by $NO_SHUTDOWN_FILE"
        return 0
    fi
    if [[ "$CONFIRM_AUTODL_SHUTDOWN" != "YES" ]]; then
        log "Automatic shutdown skipped: CONFIRM_AUTODL_SHUTDOWN is not YES"
        return 0
    fi
    log "Syncing files before AutoDL shutdown: $reason"
    sync
    sleep 5
    /usr/bin/shutdown
}

copy_to_file_storage() {
    local archive="$1"
    if [[ -d "$BACKUP_DIR" && -w "$BACKUP_DIR" ]]; then
        log "Copying result archive to $BACKUP_DIR"
        cp -f "$archive" "${archive}.sha256" "$BACKUP_DIR/"
        sync
        log "Persistent backup copied to $BACKUP_DIR"
    else
        log "File storage is not mounted/writable; archive remains in $PROJECT_DIR"
    fi
}

failure_backup() {
    local archive="schemeC_failure_$(date '+%Y%m%d_%H%M%S').tar.gz"
    tar --ignore-failed-read -czf "$archive" \
        logs configs \
        artifacts/fold0/autoencoder/summary.json \
        artifacts/fold0/autoencoder/history.jsonl \
        artifacts/fold0/autoencoder/resolved_config.json \
        artifacts/fold0/autoencoder/best.pt \
        artifacts/fold0/autoencoder/last.pt \
        artifacts/fold0/autoencoder/final.pt \
        artifacts/fold0/autoencoder/evaluation.json \
        artifacts/fold0/autoencoder/ablation.json \
        artifacts/fold0/autoencoder/quality_gate.json \
        artifacts/capacity/one_sample.json \
        artifacts/capacity/thirty_two_samples.json \
        artifacts/fold0/context/summary.json \
        artifacts/final/autoencoder/summary.json \
        artifacts/final/context/summary.json \
        2>/dev/null || true
    if [[ -s "$archive" ]]; then
        sha256sum "$archive" > "${archive}.sha256" || true
        copy_to_file_storage "$archive" || true
        log "Failure diagnostics saved to $archive"
    fi
}

on_error() {
    local status=$?
    local line="${BASH_LINENO[0]:-unknown}"
    trap - ERR
    set +e
    log "ERROR at stage=$CURRENT_STAGE line=$line status=$status"
    write_status "FAILED" "command failed at line $line with status $status"
    failure_backup
    if [[ "$SHUTDOWN_ON_FAILURE" == "1" ]]; then
        shutdown_instance "pipeline failure"
    fi
    exit "$status"
}
trap on_error ERR

run_logged() {
    log "START: $CURRENT_STAGE"
    "$@" 2>&1 | tee -a "$MASTER_LOG"
    log "DONE: $CURRENT_STAGE"
}

if [[ "$CONFIRM_AUTODL_SHUTDOWN" != "YES" ]]; then
    log "WARNING: this run will not shut down unless CONFIRM_AUTODL_SHUTDOWN=YES"
fi
if [[ ! -x /usr/bin/shutdown ]]; then
    log "ERROR: /usr/bin/shutdown is unavailable; refusing to start unattended work"
    write_status "FAILED" "shutdown command is unavailable"
    exit 2
fi

write_status "RUNNING" "overnight pipeline started in mode=$PIPELINE_MODE"
log "Scheme C overnight pipeline started in $PROJECT_DIR mode=$PIPELINE_MODE"

CURRENT_STAGE="CUDA and data preflight"
run_logged python scripts/check_environment.py \
    --config configs/fold0_5090.json --require-cuda

if [[ "$PIPELINE_MODE" == "ae" || "$PIPELINE_MODE" == "fold0" || "$PIPELINE_MODE" == "all" ]]; then
    CURRENT_STAGE="AE capacity gates"
    if python scripts/verify_completion.py --stage capacity >/dev/null 2>&1; then
        log "Existing AE capacity reports passed; skipping capacity retraining"
    else
        run_logged bash scripts/run_ae_capacity_gates.sh
    fi
    run_logged python scripts/verify_completion.py \
        --stage capacity --output artifacts/capacity/completion_report.json

fi

if [[ "$PIPELINE_MODE" == "ae" ]]; then
    CURRENT_STAGE="AE preprocessing"
    if [[ ! -f artifacts/preprocessed_scheme_c/manifest.json ]]; then
        run_logged python scripts/preprocess.py --config configs/fold0_5090.json
    else
        log "Existing preprocessing manifest found; skipping preprocessing"
    fi

    CURRENT_STAGE="AE checkpoint compatibility"
    run_logged python scripts/ensure_run_compatibility.py \
        --config configs/fold0_5090.json --run fold0

    CURRENT_STAGE="formal Fold0 AE training"
    if python scripts/verify_completion.py --stage ae >/dev/null 2>&1; then
        log "Existing formal AE analysis passed artifact verification; skipping retraining"
    else
        rm -f \
            artifacts/fold0/autoencoder/evaluation.json \
            artifacts/fold0/autoencoder/ablation.json \
            artifacts/fold0/autoencoder/quality_gate.json \
            artifacts/fold0/autoencoder/completion_report.json
        if [[ -f artifacts/fold0/autoencoder/last.pt ]]; then
            run_logged python scripts/train_autoencoder.py \
                --config configs/fold0_5090.json --resume
        else
            run_logged python scripts/train_autoencoder.py \
                --config configs/fold0_5090.json
        fi

        CURRENT_STAGE="AE full validation"
        run_logged python scripts/evaluate.py \
            --config configs/fold0_5090.json \
            --stage autoencoder \
            --checkpoint artifacts/fold0/autoencoder/best.pt \
            --output artifacts/fold0/autoencoder/evaluation.json

        CURRENT_STAGE="AE detail ablation"
        run_logged python scripts/evaluate_ae_ablation.py \
            --config configs/fold0_5090.json \
            --checkpoint artifacts/fold0/autoencoder/best.pt \
            --output artifacts/fold0/autoencoder/ablation.json

        CURRENT_STAGE="AE quality gate"
        log "START: $CURRENT_STAGE"
        gate_exit=0
        if python scripts/check_ae_gate.py \
            --config configs/fold0_5090.json \
            --evaluation artifacts/fold0/autoencoder/evaluation.json \
            --ablation artifacts/fold0/autoencoder/ablation.json \
            --output artifacts/fold0/autoencoder/quality_gate.json \
            2>&1 | tee -a "$MASTER_LOG"; then
            log "DONE: $CURRENT_STAGE (PASS)"
        else
            gate_exit=$?
            log "DONE: $CURRENT_STAGE (model gate did not pass, status=$gate_exit; packaging analysis instead of starting Context)"
        fi
    fi

    CURRENT_STAGE="AE artifact verification"
    run_logged python scripts/verify_completion.py \
        --stage ae \
        --output artifacts/fold0/autoencoder/completion_report.json

    CURRENT_STAGE="AE analysis packaging"
    run_logged bash scripts/package_ae_analysis.sh
    shopt -s nullglob
    ae_archives=(schemeC_ae_analysis_*.tar.gz)
    if (( ${#ae_archives[@]} == 0 )); then
        log "No Scheme C AE analysis archive was produced"
        false
    fi
    latest_archive="${ae_archives[0]}"
    for candidate in "${ae_archives[@]:1}"; do
        if [[ "$candidate" -nt "$latest_archive" ]]; then
            latest_archive="$candidate"
        fi
    done
    tar -tzf "$latest_archive" >/dev/null
    sha256sum -c "${latest_archive}.sha256"
    copy_to_file_storage "$latest_archive"
    gate_status="$(python -c "import json; print(json.load(open('artifacts/fold0/autoencoder/quality_gate.json'))['status'])")"

    CURRENT_STAGE="complete"
    write_status "SUCCESS" "formal AE analysis archived; quality_gate=$gate_status"
    log "SUCCESS: formal Scheme C AE analysis completed; quality_gate=$gate_status"
    log "AE analysis archive: $latest_archive"
    if [[ "$SHUTDOWN_ON_SUCCESS" == "1" ]]; then
        shutdown_instance "AE analysis pipeline complete (quality_gate=$gate_status)"
    fi
    exit 0
fi

if [[ "$PIPELINE_MODE" == "fold0" || "$PIPELINE_MODE" == "all" ]]; then
    CURRENT_STAGE="fold0 verification/training"
    if python scripts/verify_completion.py --stage fold0 >/dev/null 2>&1; then
        log "Existing Fold0 artifacts passed verification; skipping Fold0 retraining"
    else
        run_logged env RESUME=1 bash scripts/run_fold0.sh
    fi
    run_logged python scripts/verify_completion.py \
        --stage fold0 --output artifacts/fold0/completion_report.json
fi

if [[ "$PIPELINE_MODE" == "fold0" ]]; then
    CURRENT_STAGE="fold0 packaging"
    run_logged bash scripts/package_fold0.sh
    shopt -s nullglob
    fold_archives=(schemeC_fold0_*.tar.gz)
    if (( ${#fold_archives[@]} == 0 )); then
        log "No Scheme C Fold0 archive was produced"
        false
    fi
    latest_archive="${fold_archives[0]}"
    for candidate in "${fold_archives[@]:1}"; do
        if [[ "$candidate" -nt "$latest_archive" ]]; then
            latest_archive="$candidate"
        fi
    done
    tar -tzf "$latest_archive" >/dev/null
    sha256sum -c "${latest_archive}.sha256"
    copy_to_file_storage "$latest_archive"
    CURRENT_STAGE="complete"
    write_status "SUCCESS" "capacity gates and Fold0 archive verified"
    log "SUCCESS: Scheme C capacity gates and Fold0 completed"
    log "Fold0 archive: $latest_archive"
    if [[ "$SHUTDOWN_ON_SUCCESS" == "1" ]]; then
        shutdown_instance "Fold0 pipeline success"
    fi
    exit 0
fi

CURRENT_STAGE="required Fold0 verification"
run_logged python scripts/verify_completion.py \
    --stage fold0 --output artifacts/fold0/completion_report.json

CURRENT_STAGE="final config selection"
run_logged python scripts/prepare_final_config.py

CURRENT_STAGE="4000-sample final training and inference"
if python scripts/verify_completion.py --stage final >/dev/null 2>&1; then
    log "Existing final artifacts passed verification; skipping final retraining"
else
    run_logged env RESUME=1 bash scripts/run_final.sh
fi
run_logged python scripts/verify_completion.py \
    --stage final --output artifacts/final/completion_report.json

CURRENT_STAGE="result packaging"
run_logged bash scripts/package_results.sh
shopt -s nullglob
archives=(schemeC_results_*.tar.gz)
if (( ${#archives[@]} == 0 )); then
    log "No Scheme C result archive was produced"
    false
fi
latest_archive="${archives[0]}"
for candidate in "${archives[@]:1}"; do
    if [[ "$candidate" -nt "$latest_archive" ]]; then
        latest_archive="$candidate"
    fi
done
tar -tzf "$latest_archive" >/dev/null
sha256sum -c "${latest_archive}.sha256"
copy_to_file_storage "$latest_archive"

CURRENT_STAGE="complete"
write_status "SUCCESS" "final model, 500-sample test channel and archive verified"
log "SUCCESS: Scheme C final training, inference and packaging completed (mode=$PIPELINE_MODE)"
log "Final output: outputs/final/Round2_Test_Channel.npy"
log "Result archive: $latest_archive"

if [[ "$SHUTDOWN_ON_SUCCESS" == "1" ]]; then
    shutdown_instance "pipeline success"
fi
