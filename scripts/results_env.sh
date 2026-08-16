#!/bin/bash
# Results-publishing preflight, sourced by every job script that trains or
# evaluates a model:
#
#   source "$REPO_DIR/scripts/results_env.sh"
#
# Loads .env (so GITHUB_TOKEN and BLT_LCM_PUSH_RESULTS reach the python
# process), reports how this job will publish, and -- crucially -- verifies the
# push credentials NOW rather than after the job has spent eleven hours on a
# GPU. `git ls-remote` is a cheap authenticated round-trip: if it fails here,
# the push at the end would have failed too.
#
# Never fatal. A job whose results cannot be pushed should still run and still
# write its results to disk; it just says so up front.

_RESULTS_ENV_REPO=${REPO_DIR:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")}

# `set -a` exports everything the file defines, which is what gets the token
# into the python process's environment.
if [ -f "$_RESULTS_ENV_REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$_RESULTS_ENV_REPO/.env"
    set +a
fi

# Never let git block on a credential prompt: a compute node has no terminal,
# so an interactive prompt means the job sits idle on a GPU until Slurm's
# wall-clock limit kills it.
export GIT_TERMINAL_PROMPT=0

echo "Results publishing:"

if [ "${BLT_LCM_PUSH_RESULTS:-0}" != "1" ]; then
    echo "  push: OFF (results are still collected and committed locally)"
    echo "  set BLT_LCM_PUSH_RESULTS=1 in .env to push them"
else
    _rs_remote=${BLT_LCM_RESULTS_REMOTE:-origin}
    _rs_branch=${BLT_LCM_RESULTS_BRANCH:-$(git -C "$_RESULTS_ENV_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)}
    _rs_url=$(git -C "$_RESULTS_ENV_REPO" remote get-url "$_rs_remote" 2>/dev/null)
    echo "  push: ON -> $_rs_remote/${_rs_branch:-<detached>}"

    if [ -z "$_rs_url" ]; then
        echo "  WARNING: no remote named '$_rs_remote' -- results will stay local"
    else
        case "$_rs_url" in
            https://*)
                if [ -n "${GITHUB_TOKEN:-}${GIT_TOKEN:-}${BLT_LCM_GIT_TOKEN:-}" ]; then
                    echo "  auth: token from the environment (user ${GITHUB_USERNAME:-x-access-token})"
                else
                    echo "  WARNING: https remote but no GITHUB_TOKEN in the environment."
                    echo "           Add GITHUB_TOKEN to .env, or use an ssh remote."
                fi
                ;;
            *) echo "  auth: ssh key / agent" ;;
        esac

        # The actual check. Credentials are injected the same way
        # results_sync.py does it, and the URL is never echoed.
        _rs_probe_url="$_rs_url"
        _rs_token=${BLT_LCM_GIT_TOKEN:-${GIT_TOKEN:-${GITHUB_TOKEN:-}}}
        case "$_rs_url" in
            https://*)
                if [ -n "$_rs_token" ]; then
                    _rs_probe_url="https://${GITHUB_USERNAME:-x-access-token}:${_rs_token}@${_rs_url#https://}"
                fi
                ;;
        esac
        if git -C "$_RESULTS_ENV_REPO" ls-remote --heads "$_rs_probe_url" >/dev/null 2>&1; then
            echo "  credentials: OK (verified before the job starts)"
        else
            echo "  WARNING: could not authenticate to the remote."
            echo "           The job will run and results will be written to disk,"
            echo "           but the push at the end will fail. Check GITHUB_TOKEN"
            echo "           in .env (needs 'repo' scope, or fine-grained"
            echo "           'Contents: read and write' on this repository)."
        fi
        unset _rs_probe_url _rs_token
    fi
    unset _rs_remote _rs_branch _rs_url
fi

# Under a scheduler, several array jobs share one checkout. results_sync.py
# detects this and builds its commit through a private index instead of the
# shared one; say so, because it explains why HEAD does not move locally.
if [ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}" ]; then
    echo "  mode: isolated commit (shared checkout safe; local HEAD is not moved)"
fi

unset _RESULTS_ENV_REPO
