#!/bin/bash
# Check the outcome of a scripts/smoke_test.sh run: queries Slurm's own record
# of each job's exit state via `sacct`, and separately greps its log for
# common Python/CUDA failure signatures. Best run once jobs have had a chance
# to either fail fast, complete, hit their --time cutoff, or be cancelled --
# running it against still-PENDING/RUNNING jobs just reports "OK-SO-FAR".
#
# Usage:
#   scripts/check_smoke_test.sh [tag]        # reads logs/smoke_test_<tag>.jobids
#   scripts/check_smoke_test.sh galvani
#   scripts/check_smoke_test.sh --jobs 2755739,2755740,2755741
#
# Exit code is 0 if nothing was flagged FAIL, 1 otherwise.
set -uo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")
cd "$REPO_DIR"

declare -A JOB_IDS

if [ "${1:-}" = "--jobs" ]; then
    IFS=',' read -ra ids <<< "${2:-}"
    i=0
    for id in "${ids[@]}"; do
        [ -n "$id" ] || continue
        JOB_IDS["job$i"]="$id"
        i=$((i + 1))
    done
else
    TAG=${1:-smoketest}
    STATE_FILE="logs/smoke_test_${TAG}.jobids"
    if [ ! -f "$STATE_FILE" ]; then
        echo "No job-id file at $STATE_FILE -- pass a tag matching a smoke_test.sh run, or use --jobs id1,id2,..." >&2
        exit 1
    fi
    while IFS='=' read -r key id; do
        [ -n "$key" ] && JOB_IDS[$key]="$id"
    done < "$STATE_FILE"
fi

if [ "${#JOB_IDS[@]}" -eq 0 ]; then
    echo "No job IDs to check." >&2
    exit 1
fi

# Generic Python/CUDA/Slurm error signatures -- not tied to any one script's
# exact print statements, so this needs no upkeep as lcm_scripts/*.py change.
ERROR_PATTERN='Traceback \(most recent call last\)|CUDA error|CUDA out of memory|OutOfMemoryError|ModuleNotFoundError|ImportError|FATAL|srun: error|[Ee]rror:'

overall_fail=0
printf "%-16s %-10s %-14s %-14s %s\n" "JOB" "ID" "SLURM_STATE" "VERDICT" "NOTE"

for key in "${!JOB_IDS[@]}"; do
    id="${JOB_IDS[$key]}"
    state=$(sacct -j "$id" --format=State --noheader -P 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$state" ] && state="PENDING"

    logfile=$(ls logs/*_"${id}".out 2>/dev/null | head -1)
    errfile=$(ls logs/*_"${id}".err 2>/dev/null | head -1)

    hit=""
    for f in "$logfile" "$errfile"; do
        [ -n "$f" ] && [ -f "$f" ] || continue
        m=$(grep -E -m1 "$ERROR_PATTERN" "$f" 2>/dev/null)
        [ -n "$m" ] && { hit="$m"; break; }
    done

    verdict="UNKNOWN"
    note=""

    if [ -z "$logfile" ] && [ -z "$errfile" ]; then
        verdict="PENDING"
        note="no log file yet"
    else
        case "$state" in
            FAILED|NODE_FAIL|OUT_OF_MEMORY)
                verdict="FAIL"
                note="slurm=$state${hit:+; log: ${hit:0:60}}"
                ;;
            TIMEOUT)
                if [ -n "$hit" ]; then
                    verdict="FAIL"
                    note="errored before hitting the time cutoff; log: ${hit:0:60}"
                else
                    verdict="PASS?"
                    note="hit the --time cutoff with no error -- looks like it was training fine"
                fi
                ;;
            CANCELLED)
                if [ -n "$hit" ]; then
                    verdict="FAIL"
                    note="errored before being cancelled; log: ${hit:0:60}"
                else
                    verdict="PASS? (cancelled)"
                    note="no error before cancellation -- fine if you cancelled this on purpose"
                fi
                ;;
            COMPLETED)
                if [ -n "$hit" ]; then
                    verdict="FAIL"
                    note="completed but log has an error line: ${hit:0:60}"
                else
                    verdict="PASS"
                    note="completed cleanly"
                fi
                ;;
            RUNNING|PENDING|CONFIGURING|"")
                if [ -n "$hit" ]; then
                    verdict="FAIL"
                    note="log already shows an error: ${hit:0:60}"
                else
                    verdict="OK-SO-FAR"
                    note="slurm=$state, no error yet"
                fi
                ;;
            *)
                verdict="UNKNOWN"
                note="unrecognized slurm state: $state${hit:+; log: ${hit:0:60}}"
                ;;
        esac
    fi

    [ "$verdict" = "FAIL" ] && overall_fail=1

    printf "%-16s %-10s %-14s %-14s %s\n" "$key" "$id" "$state" "$verdict" "$note"
done

echo ""
if [ "$overall_fail" -eq 1 ]; then
    echo "At least one job FAILED -- see the NOTE column, then check its full log."
else
    echo "No hard failures detected. TIMEOUT/CANCELLED with no error is expected and fine for a smoke test."
fi

exit "$overall_fail"
