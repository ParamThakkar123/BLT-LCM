#!/bin/bash
# Check the outcome of a scripts/smoke_test.sh run: queries Slurm's own record
# of each job's exit state via `sacct`, and separately greps its log for
# common Python/CUDA failure signatures. Best run once jobs have had a chance
# to either fail fast, complete, hit their --time cutoff, or be cancelled --
# running it against still-PENDING/RUNNING jobs just reports "OK-SO-FAR".
#
# Usage:
#   scripts/check_smoke_test.sh            # reads logs/smoke_test.jobids
#   scripts/check_smoke_test.sh galvani    # reads logs/smoke_test_galvani.jobids
#   scripts/check_smoke_test.sh --jobs 2755739,2755740,2755741
#
# Writes the summary table to logs/smoke_test[_<tag>]_report.txt in addition
# to stdout, PLUS the last 200 lines (tqdm \r converted to \n) of both stdout
# and stderr for every job flagged FAIL, so the whole result -- including
# enough to diagnose a real failure -- can be copied off the cluster as one
# file: scp <user>@<login-node>:BLT-LCM/logs/smoke_test_report.txt .
#
# Exit code is 0 if nothing was flagged FAIL, 1 otherwise.
set -uo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")
cd "$REPO_DIR"
mkdir -p logs

declare -A JOB_IDS

if [ "${1:-}" = "--jobs" ]; then
    STATE_FILE=""
    REPORT_FILE="logs/smoke_test_jobs_report.txt"
    IFS=',' read -ra ids <<< "${2:-}"
    i=0
    for id in "${ids[@]}"; do
        [ -n "$id" ] || continue
        JOB_IDS["job$i"]="$id"
        i=$((i + 1))
    done
else
    # Only suffix filenames with the tag if one was actually passed -- an
    # implicit default tag would otherwise redundantly read
    # "smoke_test_smoketest.jobids"/"smoke_test_smoketest_report.txt".
    if [ -n "${1:-}" ]; then
        STATE_FILE="logs/smoke_test_${1}.jobids"
        REPORT_FILE="logs/smoke_test_${1}_report.txt"
    else
        STATE_FILE="logs/smoke_test.jobids"
        REPORT_FILE="logs/smoke_test_report.txt"
    fi
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
# exact print statements, so this needs no upkeep as lcm_scripts/*.py changes.
ERROR_PATTERN='Traceback \(most recent call last\)|CUDA error|CUDA out of memory|OutOfMemoryError|ModuleNotFoundError|ImportError|FATAL|srun: error|[Ee]rror:'
# Slurm's own routine job-lifecycle messages -- stripped before matching
# ERROR_PATTERN so e.g. a plain --time cutoff (which Slurm reports as
# "error: *** JOB ... CANCELLED AT ... DUE TO TIME LIMIT ***") isn't mistaken
# for a program error.
SLURM_NOISE_PATTERN='slurmstepd:|\*\*\* JOB [0-9]+ ON|DUE TO TIME LIMIT|CANCELLED AT'

# Prints up to one ERROR_PATTERN match plus ~100 trailing chars of context,
# after converting embedded \r (used by tqdm to overwrite progress bars in
# place) to real newlines first -- otherwise a whole progress bar's worth of
# updates is one giant "line" to grep, and slicing from the start of it shows
# stale tqdm noise instead of the actual match.
find_hit() {
    local file="$1"
    [ -f "$file" ] || return 0
    grep -a -v -E "$SLURM_NOISE_PATTERN" "$file" 2>/dev/null \
        | tr '\r' '\n' \
        | grep -a -m1 -E -o "($ERROR_PATTERN).{0,100}"
}

main() {
    echo "Report generated $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
    echo ""

    local overall_fail=0
    declare -A LOGFILES ERRFILES
    local failed_keys=()
    printf "%-16s %-10s %-14s %-14s %s\n" "JOB" "ID" "SLURM_STATE" "VERDICT" "NOTE"

    for key in "${!JOB_IDS[@]}"; do
        local id="${JOB_IDS[$key]}"
        local state
        state=$(sacct -j "$id" --format=State --noheader -P 2>/dev/null | head -1 | tr -d ' ')
        [ -z "$state" ] && state="PENDING"

        local logfile errfile
        logfile=$(ls logs/*_"${id}".out 2>/dev/null | head -1)
        errfile=$(ls logs/*_"${id}".err 2>/dev/null | head -1)
        LOGFILES[$key]="$logfile"
        ERRFILES[$key]="$errfile"

        local hit="" f m
        for f in "$logfile" "$errfile"; do
            [ -n "$f" ] || continue
            m=$(find_hit "$f")
            [ -n "$m" ] && { hit="$m"; break; }
        done

        local verdict="UNKNOWN" note=""

        if [ -z "$logfile" ] && [ -z "$errfile" ]; then
            verdict="PENDING"
            note="no log file yet"
        else
            case "$state" in
                FAILED|NODE_FAIL|OUT_OF_MEMORY)
                    verdict="FAIL"
                    note="slurm=$state${hit:+; log: $hit}"
                    ;;
                TIMEOUT)
                    if [ -n "$hit" ]; then
                        verdict="FAIL"
                        note="errored before hitting the time cutoff; log: $hit"
                    else
                        verdict="PASS?"
                        note="hit the --time cutoff with no error -- looks like it was training fine"
                    fi
                    ;;
                CANCELLED)
                    if [ -n "$hit" ]; then
                        verdict="FAIL"
                        note="errored before being cancelled; log: $hit"
                    else
                        verdict="PASS? (cancelled)"
                        note="no error before cancellation -- fine if you cancelled this on purpose"
                    fi
                    ;;
                COMPLETED)
                    if [ -n "$hit" ]; then
                        verdict="FAIL"
                        note="completed but log has an error line: $hit"
                    else
                        verdict="PASS"
                        note="completed cleanly"
                    fi
                    ;;
                RUNNING|PENDING|CONFIGURING|"")
                    if [ -n "$hit" ]; then
                        verdict="FAIL"
                        note="log already shows an error: $hit"
                    else
                        verdict="OK-SO-FAR"
                        note="slurm=$state, no error yet"
                    fi
                    ;;
                *)
                    verdict="UNKNOWN"
                    note="unrecognized slurm state: $state${hit:+; log: $hit}"
                    ;;
            esac
        fi

        if [ "$verdict" = "FAIL" ]; then
            overall_fail=1
            failed_keys+=("$key")
        fi

        printf "%-16s %-10s %-14s %-14s %s\n" "$key" "$id" "$state" "$verdict" "$note"
    done

    echo ""
    if [ "$overall_fail" -eq 1 ]; then
        echo "At least one job FAILED -- full stdout/stderr for each follows below."
    else
        echo "No hard failures detected. TIMEOUT/CANCELLED with no error is expected and fine for a smoke test."
    fi

    if [ "${#failed_keys[@]}" -gt 0 ]; then
        echo ""
        echo "===================================================================="
        echo "Full logs for FAILED jobs (last 200 lines each; tqdm's \\r converted"
        echo "to \\n so progress-bar output doesn't collapse into one huge line)"
        echo "===================================================================="
        for key in "${failed_keys[@]}"; do
            local id="${JOB_IDS[$key]}"
            local logfile="${LOGFILES[$key]}"
            local errfile="${ERRFILES[$key]}"
            echo ""
            echo "---- $key (job $id) ----"
            if [ -n "$logfile" ]; then
                echo "-- stdout: $logfile (last 200 lines) --"
                tr '\r' '\n' < "$logfile" | tail -n 200
            else
                echo "-- stdout: no .out file found --"
            fi
            if [ -n "$errfile" ]; then
                echo "-- stderr: $errfile (last 200 lines) --"
                tr '\r' '\n' < "$errfile" | tail -n 200
            else
                echo "-- stderr: no .err file found --"
            fi
        done
    fi

    exit "$overall_fail"
}

# Run through a pipe (not `exec > >(tee ...)`) so the exit code is captured
# via PIPESTATUS without racing tee's subshell for the last bit of output.
main | tee "$REPORT_FILE"
result=${PIPESTATUS[0]}

echo ""
echo "Full report written to: $REPORT_FILE"
echo "Copy it off the cluster with, e.g.: scp <user>@<login-node>:$(pwd)/$REPORT_FILE ."

exit "$result"
