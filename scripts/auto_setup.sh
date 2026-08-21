#!/usr/bin/env bash
#
# auto_setup.sh -- one-command setup + full experiment driver for BLT-LCM.
#
# Lives in the repo, so it runs from an existing checkout:
#
#   1. cds to the repository root (the parent of this script)
#   2. installs the CUDA 13.0 torch/torchvision wheels and runs `uv sync`
#   3. probes the GPU's VRAM and derives a batch size per stage that fits
#   4. runs the decoder/pooler training, Stage 1 encode, Stage 2 MT grid,
#      Stage 3 baselines and Stage 4 analyses -- in that order
#
# Every step writes its own log, records its artifacts, and drops a completion
# marker, so re-running the script resumes from the first step that has not
# finished instead of redoing the whole pipeline. Inside a step, the python
# scripts' own `--resume auto` checkpointing picks up mid-epoch.
#
# Resume is decided from the RESULTS, not only from the driver's own markers.
# Before a step runs, its outputs are inspected: if they are complete -- every
# requested epoch trained, and every expected artifact (checkpoint, metrics CSV,
# published run record) present -- the step is marked done and the pipeline
# continues from the first step that is genuinely unfinished. That covers a
# wiped runs/auto_setup state directory, steps that were run by hand outside the
# driver, and a kill between a step finishing and its marker being written. A
# step whose results are only partial is re-run, and its `--resume auto` picks
# up from the last checkpoint instead of restarting the epochs already paid for.
#
# Results already PUSHED count too. Every step publishes its record to
# results/runs/<run>/run.json, so the driver fetches the results refs once and
# reads them straight out of git -- no clone, no pull, no checkout -- to see
# which cells of the grid another machine has already finished. A step whose
# published record matches this step's configuration and ran all its epochs is
# not run again.
#
# With one deliberate exception: checkpoints and embedding caches are never
# published (they are large and regenerable), so a pushed record proves a step
# RAN, not that its outputs are on this machine. Steps whose files later stages
# consume -- the decoder/pooler and the Stage 1 encode -- are therefore only
# skipped on remote evidence when those files are also here. Everything else
# (the MT grid, the baselines) exists to produce the published result, and a
# published result is the whole job.
#
# Ctrl-C stops the pipeline, not just the running step: the interrupted step is
# left unmarked (so a re-run resumes it) and nothing further is started.
#
# The driver's own state -- every step's log, its completion marker, the
# artifacts it copied, the manifest -- is published too. runs/auto_setup is
# gitignored (it sits next to multi-GB scratch), so at the end of every run the
# whole directory is mirrored into results/auto_setup/<machine>/ and committed,
# and pushed when BLT_LCM_PUSH_RESULTS=1. Oversized logs go in as their tail,
# credential-shaped strings are masked first, and whatever is left out is named
# in WITHHELD.txt. See lcm_scripts/publish_state.py.
#
# A step that is already done -- because this machine finished it earlier, or
# because another machine ran it and pushed the result -- has its published
# files pulled back BEFORE that mirror is taken: the metrics CSV, the figures,
# the loss history and the record itself are extracted from the results refs
# into results/runs/<run>/ (whatever this checkout is missing) and into the
# step's artifacts directory. So the published state directory carries the
# evidence for every step of the pipeline, not only for the steps that happened
# to run here. `--publish-state` does exactly that and nothing else: restore
# everything published, mirror it, push, run no experiments.
#
# Usage (from anywhere -- the script finds its own repo root):
#   bash scripts/auto_setup.sh                  # everything, resuming as needed
#   bash scripts/auto_setup.sh --list           # show the plan + chosen batch sizes
#   bash scripts/auto_setup.sh --dry-run        # print commands, run nothing
#   bash scripts/auto_setup.sh --only 'mt:*'    # only the Stage 2 MT grid
#   bash scripts/auto_setup.sh --skip 'bench:*' # everything except the baselines
#   bash scripts/auto_setup.sh --force --only decoder
#   bash scripts/auto_setup.sh --publish-state  # collect + push the state dir only
#
# Environment knobs (all optional):
#   REPO_DIR      repository root  (default: the parent of this script)
#   STATE_DIR     progress + artifacts root   (default: $REPO_DIR/runs/auto_setup)
#   VRAM_MB       override the probed VRAM in MiB (e.g. for a shared GPU)
#   BS_DECODER / BS_ENCODE / BS_MT   override a stage's batch size outright
#   MT_ALLOW_LARGER_BATCH=1   let Stage 2 exceed the script default of 32
#   AMP_ENCODE=1  add --amp to the Stage 1 encode (off: cached embeddings stay fp32)
#   RUN_LLAMA=1   also run the Llama-8B QLoRA baseline (needs >=20 GiB + HF auth)
#   KEEP_CU130=1  pass --no-sync to `uv run` so an out-of-band torch survives (see NOTE)
#   UV_RUN_OVERRIDE  replace the `uv run` launcher entirely (apptainer, srun, tests)
#   STOP_ON_FAIL=1  abort on the first failing experiment step
#   MAX_OOM_RETRIES  halve the batch size and retry this many times (default 3)
#   ADOPT_COMPLETE=0  do not adopt already-complete results; only the driver's
#                     own markers decide what to skip
#   RESULTS_RUNS_DIR  where the python steps publish their run records
#                     (default: results/runs -- match --results_dir if changed)
#   REMOTE_RESULTS=0  ignore results pushed by other machines (local only)
#   RESULTS_REFS      refs to read published records from
#                     (default: origin/<current branch> and origin/main)
#   RESULTS_FETCH=0   use the refs already in this clone; skip the git fetch
#   REMOTE_STRICT=1   only trust a published record whose recorded commit is an
#                     ancestor of HEAD (i.e. not produced by divergent code)
#   TRUST_REMOTE_PRODUCERS=1  skip the decoder/encode steps on remote evidence
#                     even when their (unpublished) checkpoints are missing here
#   PUBLISH_STATE=0   do not mirror/commit runs/auto_setup at the end of the run
#   PUSH_STATE=1      push that mirror (default: follow BLT_LCM_PUSH_RESULTS)
#   STATE_PUBLISH_DIR where the mirror is written (default: results/auto_setup)
#   STATE_PUBLISH_ID  its per-machine subdirectory (default: this hostname)
#   STATE_MAX_MB      a log bigger than this is published as its tail (default 5)
#   RESTORE_PUBLISHED=0  do not pull a finished step's published files back here
#
# NOTE on cu130: pyproject.toml now pins `torch==2.13.0` + `torchvision==0.28.0`
# from the cu130 index, so `uv sync` -- and every later `uv run` re-sync --
# keeps the CUDA 13.0 build rather than reverting it. This matters on Blackwell
# (sm_120) parts like the RTX PRO 6000, which the old cu121 wheels have no
# kernels for. The `uv pip install ... /cu130` below is therefore a no-op
# whenever the index head still matches the pin; it stops being one once
# upstream publishes a newer torch, and the before/after report plus the
# warning at the end of install_deps will say so. Bump the pin in
# pyproject.toml (and re-run `uv lock`) when that happens -- KEEP_CU130=1 only
# masks the drift by skipping the sync entirely.

set -uo pipefail

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

ENTROPY_MODEL=${ENTROPY_MODEL:-patching_scratch/entropy_model_marathi.pt}
COMET_MODEL=${COMET_MODEL:-Unbabel/wmt22-comet-da}
FRACTIONS=(0.25 0.50 0.80)
SEEDS=(42 43 44)
NOISE_LEVELS=(0.0 0.1 0.2)
BENCH_MODELS=(bpe_transformer bpe_lcm sonar_lcm)
DATA_SEED=${DATA_SEED:-42}
MAX_OOM_RETRIES=${MAX_OOM_RETRIES:-3}
STOP_ON_FAIL=${STOP_ON_FAIL:-0}

# Epoch counts live here rather than inline in the builders: the completion
# probe below has to ask for the same number the step is launched with, and two
# copies of "10" would drift the moment one of them was edited.
DECODER_EPOCHS=${DECODER_EPOCHS:-10}
DECODER_SENTENCES=${DECODER_SENTENCES:-50000}
MT_EPOCHS=${MT_EPOCHS:-3}
BENCH_EPOCHS=${BENCH_EPOCHS:-1}
LLAMA_EPOCHS=${LLAMA_EPOCHS:-1}

ONLY_PAT=""
SKIP_PAT=""
FORCE=0
DRY_RUN=0
LIST_ONLY=0
PUBLISH_STATE_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)    ONLY_PAT="$2"; shift 2 ;;
        --skip)    SKIP_PAT="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --list)    LIST_ONLY=1; shift ;;
        # Restore what every finished step published, mirror the state
        # directory, push it -- and run no experiments. The repair path for a
        # machine whose pipeline ran before any of this was published.
        --publish-state) PUBLISH_STATE_ONLY=1; shift ;;
        # The whole leading comment block, however long it grows: everything
        # from line 2 up to the first line that is not a comment.
        -h|--help) sed -n '2,/^[^#]/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown flag: $1 (try --help)" >&2; exit 2 ;;
    esac
done

# --------------------------------------------------------------------------- #
# logging helpers
# --------------------------------------------------------------------------- #

if [[ -t 1 ]]; then
    C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'; C_RED=$'\033[31m'
    C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_OFF=$'\033[0m'
else
    C_BOLD=""; C_DIM=""; C_RED=""; C_GRN=""; C_YEL=""; C_OFF=""
fi

now()  { date +"%Y-%m-%dT%H:%M:%S%z"; }
say()  { echo "${C_BOLD}[auto-setup]${C_OFF} $*"; }
info() { echo "${C_DIM}[auto-setup]${C_OFF} $*"; }
warn() { echo "${C_YEL}[auto-setup] WARNING:${C_OFF} $*" >&2; }
die()  { echo "${C_RED}[auto-setup] FATAL:${C_OFF} $*" >&2; exit 1; }
rule() { echo "${C_DIM}--------------------------------------------------------------------${C_OFF}"; }

# --------------------------------------------------------------------------- #
# interrupt handling
# --------------------------------------------------------------------------- #

# Ctrl-C has to stop the pipeline, not just the step that is running. The
# terminal delivers SIGINT to the whole foreground process group, so this script
# and every later step get it too: without a trap, one Ctrl-C killed the running
# training and the driver then marched through all remaining steps, each dying
# in under a second and each recorded as a genuine "FAILED (exit 130)".
INTERRUPTED=0
INTERRUPTED_STEP=""
PIPELINE_START=$(date +%s)

on_signal() {
    # A second Ctrl-C during the (short) shutdown falls through to the default
    # action, so the driver can always be killed outright.
    (( INTERRUPTED )) && return 0
    INTERRUPTED=1
    echo
    warn "received SIG$1 -- stopping the pipeline (re-run to resume where it stopped)"
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

# 128 + signal number: SIGINT and SIGTERM as a child's exit status.
interrupted_rc() { (( $1 == 130 || $1 == 143 )); }

# --------------------------------------------------------------------------- #
# step 0 -- locate the repository root
# --------------------------------------------------------------------------- #

# The script ships inside the repo, so its own location is the checkout. Every
# command below uses paths relative to that root, not to $PWD.
if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR=$(cd "$REPO_DIR" 2>/dev/null && pwd) || die "REPO_DIR does not exist: ${REPO_DIR}"
else
    REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd) \
        || die "cannot resolve the repository root from ${BASH_SOURCE[0]}"
fi

[[ -f "$REPO_DIR/pyproject.toml" && -d "$REPO_DIR/lcm_scripts" ]] \
    || die "$REPO_DIR does not look like the BLT-LCM checkout (no pyproject.toml / lcm_scripts). Set REPO_DIR."

cd "$REPO_DIR" || die "cannot cd into $REPO_DIR"
say "repository root: $REPO_DIR"

STATE_DIR=${STATE_DIR:-$REPO_DIR/runs/auto_setup}
LOG_DIR="$STATE_DIR/logs"
MARK_DIR="$STATE_DIR/state"
ART_DIR="$STATE_DIR/artifacts"
MANIFEST="$STATE_DIR/manifest.jsonl"
mkdir -p "$LOG_DIR" "$MARK_DIR" "$ART_DIR" \
         "$REPO_DIR/logs" "$REPO_DIR/embeddings" "$REPO_DIR/lcm_models" \
         "$REPO_DIR/results" "$REPO_DIR/runs" 2>/dev/null

# The repo's own shell scripts source .env; do the same so HF/W&B tokens are set.
if [[ -f "$REPO_DIR/.env" ]]; then
    set -a; . "$REPO_DIR/.env"; set +a
    info "loaded .env"
fi

# Results publishing: report where every step's results will go, and verify the
# push credentials NOW. This driver runs for many hours; discovering a bad token
# at the end of it -- once per step -- is the expensive way to find out.
# Sourced (not executed) so its exports reach the python steps below.
if [[ -f "$REPO_DIR/scripts/results_env.sh" ]]; then
    # shellcheck disable=SC1091
    . "$REPO_DIR/scripts/results_env.sh"
fi

export INDIC_RESOURCES_PATH="${INDIC_RESOURCES_PATH:-$REPO_DIR/indic_nlp_resources}"
# Fragmentation is the usual cause of a late-epoch OOM at a batch size that ran
# fine for an hour. Expandable segments cost nothing and remove that failure.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --------------------------------------------------------------------------- #
# step 1 -- dependency install
# --------------------------------------------------------------------------- #

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        info "uv $(uv --version 2>/dev/null | awk '{print $2}') found"
        return 0
    fi
    say "uv not found -- installing it"
    (( DRY_RUN )) && return 0
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh || die "uv install failed"
    else
        die "neither curl nor wget available -- install uv manually"
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv still not on PATH after install"
}

torch_build() {
    uv run --no-sync python -c \
        'import torch;print(torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())' \
        2>/dev/null || echo "(torch not importable)"
}

install_deps() {
    ensure_uv
    if (( DRY_RUN )); then
        echo "  + uv venv"
        echo "  + uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
        echo "  + uv sync"
        return 0
    fi
    # A .venv built for a different interpreter (e.g. the 3.13 one uv picked
    # before .python-version pinned 3.12) silently takes the cu130 pip install
    # below, and `uv sync` then throws that whole environment away. Recreate it
    # up front instead.
    local want have vpy
    want=$(head -n1 "$REPO_DIR/.python-version" 2>/dev/null | tr -d '[:space:]' || true)
    if [[ -d "$REPO_DIR/.venv" && -n "$want" ]]; then
        vpy="$REPO_DIR/.venv/bin/python"
        [[ -x "$vpy" ]] || vpy="$REPO_DIR/.venv/Scripts/python.exe"
        have=$("$vpy" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)
        if [[ "$have" != "$want" ]]; then
            warn "existing .venv is Python ${have:-unknown}, project wants $want -- recreating"
            rm -rf "$REPO_DIR/.venv"
        fi
    fi
    # `uv pip install` needs a venv to install into; `uv sync` would create one
    # later, but the pip step runs first, so make it explicit.
    [[ -d "$REPO_DIR/.venv" ]] || uv venv || die "uv venv failed"
    uv pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu130 || die "torch install failed"
    local before after
    before=$(torch_build)
    info "torch after the cu130 install: $before"
    uv sync || die "uv sync failed"
    after=$(torch_build)
    info "torch after uv sync:           $after"
    if [[ "$before" != "$after" ]]; then
        warn "uv sync replaced the just-installed torch with the pyproject pin ($after)."
        warn "The index head has moved past the pin. Bump torch/torchvision in"
        warn "pyproject.toml and re-run 'uv lock', or set KEEP_CU130=1 to run with"
        warn "--no-sync. Both builds are cu130, so this is drift, not a downgrade."
    fi
    printf '%s\n' "$after" > "$ART_DIR/torch_build.txt"
}

# `uv run` re-syncs the venv from the lockfile on every call unless told not to.
# UV_RUN_OVERRIDE replaces the launcher entirely (apptainer wrappers, srun, and
# the script's own smoke tests).
if [[ -n "${UV_RUN_OVERRIDE:-}" ]]; then
    read -r -a UV_RUN <<< "$UV_RUN_OVERRIDE"
elif [[ "${KEEP_CU130:-0}" == "1" ]]; then
    UV_RUN=(uv run --no-sync)
else
    UV_RUN=(uv run)
fi

# --------------------------------------------------------------------------- #
# step 2 -- VRAM probe and batch-size policy
# --------------------------------------------------------------------------- #

probe_vram() {
    local mb=""
    if [[ -n "${VRAM_MB:-}" ]]; then
        echo "$VRAM_MB"; return
    fi
    # --publish-state runs nothing, so there is no batch size to choose -- and
    # the torch fallback below would re-sync the venv to answer a question this
    # invocation does not ask.
    if (( PUBLISH_STATE_ONLY )); then echo 0; return; fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Smallest GPU wins: a batch size that fits every visible device.
        mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
             | tr -d ' \r' | sort -n | head -1)
    fi
    if [[ -z "$mb" || ! "$mb" =~ ^[0-9]+$ ]]; then
        mb=$("${UV_RUN[@]}" python -c 'import torch;print(min([torch.cuda.get_device_properties(i).total_memory//(1024*1024) for i in range(torch.cuda.device_count())]) if torch.cuda.is_available() else 0)' 2>/dev/null | tr -d ' \r')
    fi
    [[ "$mb" =~ ^[0-9]+$ ]] || mb=0
    echo "$mb"
}

gpu_report() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
            --format=csv,noheader 2>/dev/null | sed 's/^/  /'
    else
        echo "  nvidia-smi not on PATH"
    fi
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)}"
}

# Batch sizes are keyed off usable VRAM. The reference commands in the runbook
# (decoder 128, encode 8, MT 32) sit in the 30-40 GiB tier; smaller cards step
# down, bigger ones step up. The OOM retry below is the backstop.
bs_decoder_for() {
    local v=$1
    if   (( v >= 76000 )); then echo 256
    elif (( v >= 40000 )); then echo 192
    elif (( v >= 30000 )); then echo 128
    elif (( v >= 22000 )); then echo 96
    elif (( v >= 15000 )); then echo 64
    elif (( v >= 11000 )); then echo 48
    elif (( v >= 7000  )); then echo 24
    elif (( v >= 5000  )); then echo 12
    elif (( v >  0     )); then echo 8
    else echo 8; fi
}

bs_encode_for() {
    local v=$1
    if   (( v >= 76000 )); then echo 48
    elif (( v >= 40000 )); then echo 32
    elif (( v >= 30000 )); then echo 24
    elif (( v >= 22000 )); then echo 16
    elif (( v >= 15000 )); then echo 12
    elif (( v >= 11000 )); then echo 8
    elif (( v >= 7000  )); then echo 6
    elif (( v >= 5000  )); then echo 4
    elif (( v >  0     )); then echo 2
    else echo 2; fi
}

# Stage 2 produces the paper's headline numbers, so the default is capped at the
# script's own default of 32: dropping below it is crash avoidance, going above
# it changes the result. MT_ALLOW_LARGER_BATCH=1 lifts the cap for exploration.
bs_mt_for() {
    local v=$1
    if [[ "${MT_ALLOW_LARGER_BATCH:-0}" == "1" ]]; then
        if   (( v >= 76000 )); then echo 96; return
        elif (( v >= 40000 )); then echo 64; return
        fi
    fi
    if   (( v >= 22000 )); then echo 32
    elif (( v >= 15000 )); then echo 24
    elif (( v >= 11000 )); then echo 16
    elif (( v >= 7000  )); then echo 8
    elif (( v >= 5000  )); then echo 4
    elif (( v >  0     )); then echo 2
    else echo 4; fi
}

VRAM=$(probe_vram)
BS_DECODER=${BS_DECODER:-$(bs_decoder_for "$VRAM")}
BS_ENCODE=${BS_ENCODE:-$(bs_encode_for "$VRAM")}
BS_MT=${BS_MT:-$(bs_mt_for "$VRAM")}

rule
if (( PUBLISH_STATE_ONLY )); then
    say "--publish-state: restore what is published, mirror $STATE_DIR, push it"
    say "no experiment runs, and no GPU is touched"
else
    say "GPU allocation:"
    gpu_report
    if (( VRAM == 0 )); then
        warn "no CUDA GPU visible -- everything below will run on CPU and be far too"
        warn "slow for the training stages. Only Stage 4 is genuinely CPU-friendly."
    else
        say "usable VRAM: ${VRAM} MiB (smallest visible device)"
    fi
    say "batch sizes: decoder=${BS_DECODER}  encode=${BS_ENCODE}  mt=${BS_MT}"
fi
rule

# The probe record describes a run that is about to train something. A
# --publish-state invocation is not one, and would overwrite the record of the
# run whose state it is publishing with an empty one.
if (( ! DRY_RUN )) && (( ! PUBLISH_STATE_ONLY )); then
    {
        printf '{"timestamp":"%s","vram_mb":%s,"bs_decoder":%s,"bs_encode":%s,"bs_mt":%s}\n' \
            "$(now)" "$VRAM" "$BS_DECODER" "$BS_ENCODE" "$BS_MT"
    } > "$ART_DIR/gpu_probe.json"
    gpu_report > "$ART_DIR/gpu_report.txt" 2>&1
fi

# --------------------------------------------------------------------------- #
# results-based completion probe
# --------------------------------------------------------------------------- #
#
# The driver's `.done` markers live under $STATE_DIR, so anything that loses
# them -- a cleared runs/auto_setup, a step run by hand, a kill in the window
# between a step finishing and its marker being written -- made the pipeline
# redo finished work from scratch. So before running a step, ask the results
# themselves whether it is already done: every requested epoch trained and every
# expected output present. A step that passes is adopted (marked done, skipped),
# and the pipeline resumes from the first step whose results are NOT complete --
# where the python script's own `--resume auto` continues from its checkpoint.
#
# Two independent records answer "how many epochs ran", and either is enough:
#
#   <plot_dir>/<run>_history.json   TrainingHistory -- one row per epoch, written
#                                   at the end of every epoch and carried across
#                                   resumes
#   results/runs/<run>/run.json     ResultsRecorder -- `metrics.epochs_run`, and
#                                   only written when a script reaches its end
#
# The probe reads JSON and CSV only. It needs a plain python 3, never torch, so
# it costs milliseconds per step and does not touch the GPU or re-sync the venv.

ADOPT_COMPLETE=${ADOPT_COMPLETE:-1}
RESULTS_RUNS_DIR=${RESULTS_RUNS_DIR:-results/runs}

PROBE_PY=()
find_probe_python() {
    (( ${#PROBE_PY[@]} )) && return 0
    local c
    for c in "$REPO_DIR/.venv/Scripts/python.exe" "$REPO_DIR/.venv/bin/python" \
             python3 python; do
        command -v "$c" >/dev/null 2>&1 || continue
        # A name on PATH is not necessarily a working interpreter (Windows ships
        # a `python` stub that only opens the store), so prove it runs.
        "$c" -c 'import csv, json, os, sys' >/dev/null 2>&1 || continue
        PROBE_PY=("$c")
        return 0
    done
    # Nothing standalone (fresh checkout, venv not built yet): fall back to the
    # launcher the steps themselves use. Slower, but always available.
    PROBE_PY=("${UV_RUN[@]}" python)
}

# Resolved once, here: every probe below runs inside a command substitution --
# a subshell -- so a lazy lookup would re-discover the interpreter per step and
# the cache would never survive.
find_probe_python
info "results probe: ${PROBE_PY[*]}"

# results_complete <spec...> -- exit 0 when this step's results are complete.
# Prints one line either way: the evidence, or what is still missing.
#
#   --epochs N      epochs the step is asked to train (0: not a training step)
#   --history PATH  TrainingHistory sidecar to count finished epochs in
#   --run-json PATH published run record to count finished epochs in
#   --file PATH     an output that must exist and be non-empty (repeatable)
#   --csv PATH      a metrics CSV that must exist ...
#   --csv-rows N    ... and carry at least N data rows
#   --expect K=V    a hyperparameter the record must carry (repeatable). A
#                   record that disagrees describes a different run and is not
#                   counted as evidence -- which is what makes a PUBLISHED
#                   record safe to trust: same name, same settings, or nothing.
results_complete() {
    find_probe_python
    "${PROBE_PY[@]}" - "$@" <<'PY' 2>/dev/null
import csv, json, os, sys

need_epochs = 0
histories, records, files = [], [], []
csv_path, csv_rows = None, 0
expect = []

argv = sys.argv[1:]
i = 0
while i < len(argv):
    flag, value = argv[i], argv[i + 1] if i + 1 < len(argv) else ""
    if flag == "--epochs":
        need_epochs = int(value or 0)
    elif flag == "--history":
        histories.append(value)
    elif flag == "--run-json":
        records.append(value)
    elif flag == "--file":
        files.append(value)
    elif flag == "--csv":
        csv_path = value
    elif flag == "--csv-rows":
        csv_rows = int(value or 0)
    elif flag == "--expect":
        key, _, want = value.partition("=")
        expect.append((key, want))
    else:
        print("unknown probe flag %s" % flag)
        sys.exit(2)
    i += 2


def incomplete(message):
    print(message)
    sys.exit(1)


def load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def same(want, got):
    """Compare a command-line value with a hyperparameter from a record.

    Everything arrives from bash as text, so 0.5 has to match 0.50 and the
    noise levels have to match [0.0, 0.1, 0.2] -- string equality alone would
    reject records that describe exactly this run.
    """
    if isinstance(got, (list, tuple)):
        parts = [p for p in str(want).replace(",", " ").split() if p]
        if len(parts) != len(got):
            return False
        return all(same(p, g) for p, g in zip(parts, got))
    try:
        return abs(float(want) - float(got)) < 1e-9
    except (TypeError, ValueError):
        return str(want) == str(got)


def config_mismatch(blob):
    """The first expectation this record does not meet, if any."""
    hyper = blob.get("hyperparameters") or {}
    for key, want in expect:
        if key not in hyper:
            continue          # older records predate the flag; not a conflict
        if not same(want, hyper[key]):
            return "%s=%s (expected %s)" % (key, hyper[key], want)
    return None


# Epochs actually trained, taking the best evidence available. `epochs_run` and
# the history both count across resumes, so a run that was stopped and continued
# reports the total rather than what its final process happened to do. Checked
# before the artifacts, because "2/3 epochs trained" says more about what a
# re-run will do than the name of the file that is not there yet.
done = 0
plateaued = False
mismatch = None
for path in histories:
    for record in (load_json(path) or {}).get("epochs") or []:
        try:
            done = max(done, int(record.get("epoch", 0)))
        except (TypeError, ValueError):
            pass
for path in records:
    blob = load_json(path)
    if not blob:
        continue
    reason = config_mismatch(blob)
    if reason:
        # Same run name, different settings. Counting its epochs would let one
        # configuration's results stand in for another's.
        mismatch = mismatch or reason
        continue
    metrics = blob.get("metrics") or {}
    info = blob.get("info") or {}
    for value in (metrics.get("epochs_run"), info.get("epochs_observed")):
        try:
            done = max(done, int(value))
        except (TypeError, ValueError):
            pass
    # --train_until_plateau ends a run below its epoch cap on purpose. That is a
    # finished run, not a truncated one, so it must not be restarted forever.
    if info.get("mode") == "plateau" and str(info.get("stop_reason", "")).startswith(
        "plateau"
    ):
        plateaued = True

if need_epochs > 0 and done < need_epochs and not plateaued:
    if mismatch:
        incomplete("record is for a different configuration: %s" % mismatch)
    incomplete("%d/%d epochs trained" % (done, need_epochs))

# All the epochs ran; the outputs of those epochs have to be there too. A
# training step that trained but was killed before writing its metrics is not a
# result, and a re-run of it is cheap -- the checkpoint carries every epoch.
for path in files:
    if not os.path.exists(path):
        incomplete("missing %s" % path)
    if os.path.isdir(path):
        if not os.listdir(path):
            incomplete("empty directory %s" % path)
    elif os.path.getsize(path) == 0:
        incomplete("empty %s" % path)

if csv_path:
    if not os.path.exists(csv_path):
        incomplete("missing %s" % csv_path)
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    except Exception as exc:
        incomplete("unreadable %s: %s" % (csv_path, exc))
    have = max(len(rows) - 1, 0)          # minus the header row
    if have < csv_rows:
        incomplete("%s has %d/%d result rows" % (csv_path, have, csv_rows))

if need_epochs <= 0:
    print("expected outputs present")
elif plateaued:
    print("%d epochs trained (stopped at plateau); outputs present" % done)
else:
    print("%d/%d epochs trained; outputs present" % (done, need_epochs))
PY
}

# --------------------------------------------------------------------------- #
# results already pushed by another machine
# --------------------------------------------------------------------------- #
#
# Every step publishes results/runs/<run>/run.json, so a grid cell somebody else
# already finished is knowable without running anything: fetch the results refs
# once, then read the records straight out of git. `git show <ref>:<path>` reads
# a blob from a ref -- no clone, no pull, no checkout, nothing touched in the
# working tree, so this is safe to do while other jobs are running.
#
# A published record is trusted only when it describes THIS step: the run name
# carries the grid coordinates (lcm_blt_mt_fraction0.5_s43), and --expect checks
# the hyperparameters that the name does not cover. Anything else is ignored
# rather than believed.
#
# What a published record does NOT prove is that the step's files are here.
# Checkpoints and embedding caches are never published, so `decoder` finishing
# on another machine leaves this one without lcm_models/blt_decoder.pth -- which
# Stage 2 needs. Those steps are marked "producer" below and are only skipped on
# remote evidence when their outputs are also on this disk.

REMOTE_RESULTS=${REMOTE_RESULTS:-1}
RESULTS_FETCH=${RESULTS_FETCH:-1}
REMOTE_STRICT=${REMOTE_STRICT:-0}
TRUST_REMOTE_PRODUCERS=${TRUST_REMOTE_PRODUCERS:-0}
REMOTE_DIR="$STATE_DIR/remote"
RESULTS_REFS_LIST=()

init_remote_results() {
    (( REMOTE_RESULTS )) || return 0
    if ! git -C "$REPO_DIR" rev-parse --git-dir >/dev/null 2>&1; then
        REMOTE_RESULTS=0
        info "published results: not a git checkout; using local results only"
        return 0
    fi

    local branch refs=() ref
    branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [[ -n "${RESULTS_REFS:-}" ]]; then
        read -r -a refs <<< "$RESULTS_REFS"
    else
        refs=("origin/${branch:-main}" "origin/main")
    fi

    # One fetch for the whole pipeline. GIT_TERMINAL_PROMPT=0 so a missing
    # credential fails in seconds instead of blocking the driver on a prompt
    # that a batch job has no terminal to answer.
    if (( RESULTS_FETCH )); then
        local remote_branches=() b have seen
        for ref in "${refs[@]}"; do
            [[ "$ref" == origin/* ]] || continue
            b="${ref#origin/}"
            seen=0
            for have in "${remote_branches[@]:-}"; do
                [[ "$have" == "$b" ]] && seen=1
            done
            (( seen )) || remote_branches+=("$b")
        done
        if (( ${#remote_branches[@]} )); then
            if GIT_TERMINAL_PROMPT=0 git -C "$REPO_DIR" fetch --quiet origin \
                   "${remote_branches[@]}" 2>/dev/null; then
                info "published results: fetched ${remote_branches[*]} from origin"
            else
                warn "could not fetch from origin; reading the refs already in this clone"
            fi
        fi
    fi

    for ref in "${refs[@]}"; do
        git -C "$REPO_DIR" rev-parse --verify --quiet "$ref^{commit}" >/dev/null 2>&1 \
            || continue
        # De-duplicate: origin/main is in the default list twice on main.
        local seen=0 have
        for have in "${RESULTS_REFS_LIST[@]:-}"; do
            [[ "$have" == "$ref" ]] && seen=1
        done
        (( seen )) || RESULTS_REFS_LIST+=("$ref")
    done

    if (( ${#RESULTS_REFS_LIST[@]} )); then
        say "published results: reading ${RESULTS_REFS_LIST[*]}"
    else
        info "published results: no usable refs; using local results only"
        REMOTE_RESULTS=0
    fi
}

# json_field <file> <key> -- one scalar string field out of a run record.
# Deliberately not a JSON parse: this runs before the probe and only ever reads
# two flat, quoted fields the recorder writes itself.
json_field() {
    sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$1" 2>/dev/null | head -1
}

# remote_record <run_name> <dest_dir> [published_csv]
# Extracts the newest published record for <run_name> into <dest_dir> and
# prints the ref it came from. Empty output means nothing is published.
remote_record() {
    local run="$1" dest="$2" csv="${3:-}"
    local ref tmp finished best_ref="" best_time="" best_file=""
    mkdir -p "$dest" 2>/dev/null
    for ref in "${RESULTS_REFS_LIST[@]:-}"; do
        tmp="$dest/run.$(slug "$ref").json"
        git -C "$REPO_DIR" show "$ref:$RESULTS_RUNS_DIR/$run/run.json" > "$tmp" 2>/dev/null \
            || { rm -f "$tmp"; continue; }
        # ISO-8601 UTC, so the newest record is the lexicographic maximum.
        finished=$(json_field "$tmp" finished_utc)
        if [[ -z "$best_time" || "$finished" > "$best_time" ]]; then
            best_time="$finished"; best_ref="$ref"; best_file="$tmp"
        fi
    done
    [[ -n "$best_file" ]] || return 1

    cp -f "$best_file" "$dest/run.json" 2>/dev/null || return 1
    [[ -n "$csv" ]] && \
        git -C "$REPO_DIR" show "$best_ref:$RESULTS_RUNS_DIR/$run/$csv" > "$dest/$csv" 2>/dev/null
    printf '%s' "$best_ref"
}

# A record produced by code that this checkout does not contain describes a
# different program. Off by default -- results are usually pushed from a branch
# this machine has not merged -- and REMOTE_STRICT=1 turns it into a rule.
remote_code_is_ours() {
    local record="$1" commit
    (( REMOTE_STRICT )) || return 0
    commit=$(json_field "$record" git_commit)
    [[ -n "$commit" ]] || return 1
    git -C "$REPO_DIR" merge-base --is-ancestor "$commit" HEAD 2>/dev/null
}

# One fetch and one ref list for the whole pipeline, resolved here so the plan
# printed by --list already reflects what other machines have published.
init_remote_results

# --------------------------------------------------------------------------- #
# pulling a finished step's published files back onto this machine
# --------------------------------------------------------------------------- #
#
# A step can be finished without leaving anything here: another machine ran the
# cell and pushed it, and this driver skipped it on that evidence. The record
# proving that lives in the results refs, and so do the files it was published
# with -- the metrics CSV, the loss history, the figures, the README. Reading
# them back costs one `git show` per file and no network round-trip beyond the
# fetch that already happened, so a step that is done anywhere becomes a step
# whose results are HERE:
#
#   results/runs/<run>/       whatever this checkout does not already have, so
#                             Stage 4 and the paper figures can read it
#   $ART_DIR/<step>/          the same files as that step's artifacts, so the
#                             state directory this driver publishes carries the
#                             evidence for the whole pipeline and not just for
#                             the steps that happened to run on this machine
#
# Files this machine produced itself are never overwritten: the working tree
# wins, and only the gaps are filled.

RESTORE_PUBLISHED=${RESTORE_PUBLISHED:-1}

restore_published() {
    local id="$1" run="$2" ref="${3:-}"
    (( RESTORE_PUBLISHED )) || return 0
    (( REMOTE_RESULTS ))    || return 0
    [[ -n "$run" ]]         || return 0
    (( DRY_RUN || LIST_ONLY )) && return 0

    local dest stamp
    dest="$ART_DIR/$(slug "$id")"
    stamp="$dest/published_from.txt"
    # Once per (step, run): re-extracting the same blobs on every invocation
    # would cost a git process per published file per step, for nothing.
    if [[ -f "$stamp" ]] && grep -qx "run $run" "$stamp" 2>/dev/null && (( ! FORCE )); then
        return 0
    fi

    if [[ -z "$ref" ]]; then
        ref=$(remote_record "$run" "$REMOTE_DIR/$(slug "$id")") || return 0
        [[ -n "$ref" ]] || return 0
    fi

    local prefix="$RESULTS_RUNS_DIR/$run"
    local file name got=0 filled=0 names=()
    mkdir -p "$dest" 2>/dev/null
    # -z, read straight from the pipe: ls-tree quotes unusual names in its
    # default output, and a NUL-separated list cannot be held in a variable at
    # all -- bash drops the separators in a command substitution.
    while IFS= read -r -d '' file; do
        [[ -n "$file" ]] || continue
        name="${file#"$prefix"/}"
        # The run directories are flat. A nested path would be a run inside a
        # run, and writing it under $dest by basename would collide.
        [[ "$name" == */* ]] && continue
        if git -C "$REPO_DIR" show "$ref:$file" > "$dest/$name" 2>/dev/null; then
            got=$(( got + 1 )); names+=("$name")
        else
            rm -f "$dest/$name" 2>/dev/null
            continue
        fi
        if [[ ! -e "$REPO_DIR/$file" ]]; then
            mkdir -p "$REPO_DIR/$prefix" 2>/dev/null
            git -C "$REPO_DIR" show "$ref:$file" > "$REPO_DIR/$file" 2>/dev/null \
                && filled=$(( filled + 1 )) || rm -f "$REPO_DIR/$file" 2>/dev/null
        fi
    done < <(git -C "$REPO_DIR" ls-tree -r -z --name-only "$ref" -- "$prefix/" 2>/dev/null)

    (( got )) || return 0
    {
        echo "run $run"
        echo "ref $ref"
        echo "restored $(now)"
        printf '  %s\n' "${names[@]}"
    } > "$stamp"
    info "$id -- restored $got published file(s) from $ref"
    (( filled )) && \
        info "     $filled of them into $prefix/ (this checkout did not have them)"
    return 0
}

# --------------------------------------------------------------------------- #
# publishing this driver's own state directory
# --------------------------------------------------------------------------- #
#
# runs/auto_setup holds the only record of HOW the pipeline ran -- the logs, the
# markers, the artifact lists, the manifest -- and `runs/` is gitignored, so all
# of it has always died with the machine. lcm_scripts/publish_state.py mirrors
# it into results/auto_setup/<machine>/ and commits it; pushing follows
# BLT_LCM_PUSH_RESULTS, the same switch every training job publishes under.
#
# The mirror is per-machine and is deliberately NOT the live directory: a
# `.done` marker pulled in from somebody else's run would make this driver skip
# a step whose checkpoints do not exist here.

PUBLISH_STATE=${PUBLISH_STATE:-1}
PUSH_STATE=${PUSH_STATE:-${BLT_LCM_PUSH_RESULTS:-0}}
STATE_PUBLISH_DIR=${STATE_PUBLISH_DIR:-results/auto_setup}
STATE_MAX_MB=${STATE_MAX_MB:-5}

publish_state() {
    (( PUBLISH_STATE )) || return 0
    (( DRY_RUN || LIST_ONLY )) && return 0
    local script="$REPO_DIR/lcm_scripts/publish_state.py"
    if [[ ! -f "$script" ]]; then
        info "state publishing: $script not in this checkout; skipping"
        return 0
    fi

    local cmd=("${PROBE_PY[@]}" "$script"
               --state_dir "$STATE_DIR" --repo_root "$REPO_DIR"
               --dest "$STATE_PUBLISH_DIR" --max_mb "$STATE_MAX_MB")
    [[ -n "${STATE_PUBLISH_ID:-}" ]] && cmd+=(--name "$STATE_PUBLISH_ID")
    [[ "$PUSH_STATE" == "1" ]] && cmd+=(--push)

    rule
    say "publishing the driver state directory"
    # Never fatal: a pipeline that finished its experiments must not report
    # failure because a bookkeeping commit could not be made.
    "${cmd[@]}" || warn "state publishing failed (exit $?) -- $STATE_DIR is still on disk"
    return 0
}

# --------------------------------------------------------------------------- #
# step runner: logging, resume markers, artifact capture, OOM backoff
# --------------------------------------------------------------------------- #

CMD=()               # set by each step's builder function
STEP_ARTIFACTS=()    # paths a step is expected to produce
STEP_PROBE=()        # results_complete spec; empty = no results probe
STEP_REMOTE_RUN=""   # published run name; empty = no published evidence
STEP_REMOTE_CSV=""   # metrics CSV the run publishes into its results directory
STEP_REMOTE_SPEC=()  # results_complete spec applied to the published record
STEP_KIND="terminal" # "producer": later steps consume files it does not publish
REMOTE_WHY=""        # set by remote_complete, for the log line
REMOTE_REF=""
FAILED_STEPS=()
DONE_STEPS=()
SKIPPED_STEPS=()

selected() {
    local id="$1"
    [[ -n "$ONLY_PAT" && ! "$id" == $ONLY_PAT ]] && return 1
    [[ -n "$SKIP_PAT" &&   "$id" == $SKIP_PAT ]] && return 1
    return 0
}

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# ':' and '*' are legal in a step id but not in a Windows filename.
slug() { printf '%s' "${1//[^A-Za-z0-9._-]/_}"; }

record_artifacts() {
    local id="$1" dest
    dest="$ART_DIR/$(slug "$id")"
    local out="" p sz
    mkdir -p "$dest"
    for p in "${STEP_ARTIFACTS[@]:-}"; do
        [[ -z "$p" ]] && continue
        if [[ -e "$p" ]]; then
            sz=$(wc -c < "$p" 2>/dev/null | tr -d ' ')
            out+="$p ($sz bytes)"$'\n'
            # Small, citable outputs are copied so a later run cannot clobber
            # them; multi-GB caches and checkpoints are recorded by path only.
            if [[ "$p" =~ \.(csv|json|md|txt|png|pdf)$ ]] && (( ${sz:-0} < 52428800 )); then
                cp -f "$p" "$dest/" 2>/dev/null
            fi
        else
            out+="$p (MISSING)"$'\n'
        fi
    done
    printf '%s' "$out" > "$dest/artifacts.txt"
    printf '%s' "$out"
}

# Every file this step is expected to leave on THIS machine.
artifacts_present() {
    local p
    for p in "${STEP_ARTIFACTS[@]:-}"; do
        [[ -z "$p" ]] && continue
        [[ -e "$p" ]] || return 1
    done
    return 0
}

# remote_complete <id> -- 0 when a pushed record shows this step already ran to
# completion. Sets REMOTE_REF (where the record came from) and REMOTE_WHY (the
# evidence, or why the record could not be used).
remote_complete() {
    local id="$1" dest ref why spec=()
    REMOTE_WHY=""; REMOTE_REF=""
    (( REMOTE_RESULTS )) || return 1
    [[ -n "$STEP_REMOTE_RUN" ]] || return 1

    dest="$REMOTE_DIR/$(slug "$id")"
    rm -rf "$dest" 2>/dev/null
    ref=$(remote_record "$STEP_REMOTE_RUN" "$dest" "$STEP_REMOTE_CSV") || return 1
    [[ -n "$ref" ]] || return 1
    REMOTE_REF="$ref"

    if ! remote_code_is_ours "$dest/run.json"; then
        REMOTE_WHY="REMOTE_STRICT=1 and its commit is not an ancestor of HEAD"
        return 1
    fi

    spec=(--run-json "$dest/run.json")
    (( ${#STEP_REMOTE_SPEC[@]} )) && spec+=("${STEP_REMOTE_SPEC[@]}")
    if [[ -n "$STEP_REMOTE_CSV" ]]; then
        # The metrics CSV is published alongside the record, so its absence
        # means the run ended before it wrote one.
        if [[ -s "$dest/$STEP_REMOTE_CSV" ]]; then
            spec+=(--csv "$dest/$STEP_REMOTE_CSV")
        else
            REMOTE_WHY="the published record carries no $STEP_REMOTE_CSV"
            return 1
        fi
    fi

    why=$(results_complete "${spec[@]}") || { REMOTE_WHY="$why"; return 1; }
    REMOTE_WHY="$why"
    return 0
}

# adopt_step <id> <desc> <bs> <marker> <log> <headline> <detail>
# Record a step as done without running it, because its results already exist.
# Shared by the on-disk and the published evidence paths so both leave the same
# marker, the same manifest line and the same --list output.
adopt_step() {
    local id="$1" desc="$2" bs="$3" marker="$4" log="$5" headline="$6" detail="$7"
    if (( LIST_ONLY )); then
        printf '  %-28s %s\n' "$id" "$desc"
        printf '      %s\n' "SKIP -- $headline ($detail)"
        return 0
    fi
    say "${C_GRN}complete${C_OFF} $id -- $headline ($detail)"
    SKIPPED_STEPS+=("$id ($headline)")
    (( DRY_RUN )) && return 0
    local arts
    arts=$(record_artifacts "$id")
    {
        echo "adopted $(now): $headline ($detail)"
        echo "cmd: ${CMD[*]}"
        echo "artifacts:"
        printf '%s' "$arts" | sed 's/^/  /'
    } > "$marker"
    printf '{"id":"%s","desc":"%s","status":"already_complete","evidence":"%s","detail":"%s","exit_code":0,"batch_size":"%s","elapsed_s":0,"finished":"%s","log":"%s"}\n' \
        "$(json_escape "$id")" "$(json_escape "$desc")" "$(json_escape "$headline")" \
        "$(json_escape "$detail")" "${bs:-}" "$(now)" "$(json_escape "$log")" >> "$MANIFEST"
}

run_step() {
    # run_step <id> <description> <builder-fn> [batch-size]
    local id="$1" desc="$2" builder="$3" bs="${4:-}"
    local marker="$MARK_DIR/$(slug "$id").done"
    local log="$LOG_DIR/$(slug "$id").log"

    # A signal that arrived between steps (or while one was finishing cleanly)
    # must not start the next one.
    if (( INTERRUPTED )); then
        print_summary
        exit 130
    fi

    if ! selected "$id"; then
        SKIPPED_STEPS+=("$id (filtered)")
        return 0
    fi

    # Cleared before every builder call: a builder that sets none of these must
    # not inherit the previous step's artifacts, probe or published run name.
    # The builder only assigns variables, so it is called before the marker
    # check too -- a step that is already done still has to name the run it
    # published under for its files to be restored.
    STEP_ARTIFACTS=()
    STEP_PROBE=()
    STEP_REMOTE_RUN=""
    STEP_REMOTE_CSV=""
    STEP_REMOTE_SPEC=()
    STEP_KIND="terminal"
    "$builder" "$bs"

    if [[ -f "$marker" && $FORCE -eq 0 ]]; then
        info "skip ${C_BOLD}$id${C_OFF} -- already done ($(head -1 "$marker" 2>/dev/null))"
        # Done is not the same as "its files are here": a step adopted from a
        # published record left a marker and nothing else. Fill in what the
        # results refs have, so the state directory published below carries it.
        restore_published "$id" "$STEP_REMOTE_RUN"
        SKIPPED_STEPS+=("$id (done)")
        return 0
    fi

    if (( PUBLISH_STATE_ONLY )); then
        # --publish-state: collect the evidence, run nothing. A step nobody has
        # finished yet simply has nothing to restore.
        restore_published "$id" "$STEP_REMOTE_RUN"
        SKIPPED_STEPS+=("$id (--publish-state)")
        return 0
    fi

    # No marker, but the results may still be complete -- from a run whose state
    # directory is gone, a step run by hand, or a kill between a step finishing
    # and its marker landing. Adopt those instead of recomputing them; anything
    # partial falls through and is re-run from its own checkpoint.
    local why=""
    if (( ADOPT_COMPLETE )) && (( FORCE == 0 )) && (( ${#STEP_PROBE[@]} )); then
        if why=$(results_complete "${STEP_PROBE[@]}"); then
            adopt_step "$id" "$desc" "$bs" "$marker" "$log" \
                       "results already on disk" "$why"
            return 0
        fi
        [[ -n "$why" ]] && info "$id -- results incomplete here ($why)"
    fi

    # Nothing usable on this disk -- but another machine may have finished this
    # exact cell and pushed it.
    if (( ADOPT_COMPLETE )) && (( FORCE == 0 )) && [[ -n "$STEP_REMOTE_RUN" ]]; then
        if remote_complete "$id"; then
            # Whatever it was published with belongs here now, whether or not
            # the step below still has to run for its checkpoints.
            restore_published "$id" "$STEP_REMOTE_RUN" "$REMOTE_REF"
            if [[ "$STEP_KIND" == "producer" ]] && (( ! TRUST_REMOTE_PRODUCERS )) \
               && ! artifacts_present; then
                # The record proves the step ran, not that its outputs are here:
                # checkpoints and caches are never published. Later stages read
                # those files, so this one has to run anyway.
                info "$id -- $REMOTE_REF has it ($REMOTE_WHY), but its checkpoints are"
                info "     never published and are missing here; running it so the"
                info "     later stages have them (TRUST_REMOTE_PRODUCERS=1 to skip)"
            else
                adopt_step "$id" "$desc" "$bs" "$marker" "$log" \
                           "published on $REMOTE_REF" "$REMOTE_WHY"
                return 0
            fi
        elif [[ -n "$REMOTE_WHY" ]]; then
            info "$id -- published record on ${REMOTE_REF:-origin} is not usable" \
                 "($REMOTE_WHY)"
        fi
    fi

    if (( LIST_ONLY )); then
        printf '  %-28s %s\n' "$id" "$desc"
        printf '      %s\n' "${CMD[*]}"
        return 0
    fi

    rule
    say "${C_BOLD}$id${C_OFF} -- $desc"
    info "log: $log"
    printf '      %s\n' "${CMD[*]}"

    if (( DRY_RUN )); then
        return 0
    fi

    local attempt=0 rc=0 start elapsed
    while : ; do
        start=$(date +%s)
        {
            echo "### $(now)  attempt=$attempt  batch_size=${bs:-n/a}"
            echo "### cmd: ${CMD[*]}"
        } >> "$log"
        "${CMD[@]}" 2>&1 | tee -a "$log"
        rc=${PIPESTATUS[0]}
        elapsed=$(( $(date +%s) - start ))

        if (( rc == 0 )); then break; fi

        # Killed by a signal rather than failing on its own: stop here. The OOM
        # backoff below must not "retry" an interrupted step, and the pipeline
        # must not continue into steps the same signal would kill instantly.
        if (( INTERRUPTED )) || interrupted_rc "$rc"; then
            INTERRUPTED=1
            break
        fi

        # Out of memory -> halve the batch size and try again, if the step has one.
        if [[ -n "$bs" ]] && (( attempt < MAX_OOM_RETRIES )) && (( bs > 1 )) \
           && tail -400 "$log" | grep -qiE 'out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory'; then
            attempt=$(( attempt + 1 ))
            bs=$(( bs / 2 )); (( bs < 1 )) && bs=1
            warn "$id hit CUDA OOM -- retrying at batch_size=$bs (attempt $attempt/$MAX_OOM_RETRIES)"
            "$builder" "$bs"
            continue
        fi
        break
    done

    if (( INTERRUPTED )); then
        # No completion marker and no failure entry: the step neither finished
        # nor failed, so a re-run picks it up again from its own --resume state.
        INTERRUPTED_STEP="$id"
        printf '{"id":"%s","desc":"%s","status":"interrupted","exit_code":%d,"batch_size":"%s","elapsed_s":%d,"finished":"%s","log":"%s"}\n' \
            "$(json_escape "$id")" "$(json_escape "$desc")" \
            "$rc" "${bs:-}" "$elapsed" "$(now)" "$(json_escape "$log")" >> "$MANIFEST"
        warn "$id INTERRUPTED after ${elapsed}s -- not marked done; re-run to resume it"
        print_summary
        exit 130
    fi

    local arts
    arts=$(record_artifacts "$id")

    printf '{"id":"%s","desc":"%s","status":"%s","exit_code":%d,"batch_size":"%s","elapsed_s":%d,"finished":"%s","log":"%s"}\n' \
        "$(json_escape "$id")" "$(json_escape "$desc")" \
        "$( ((rc==0)) && echo ok || echo failed )" "$rc" "${bs:-}" "$elapsed" "$(now)" \
        "$(json_escape "$log")" >> "$MANIFEST"

    if (( rc == 0 )); then
        {
            echo "finished $(now) in ${elapsed}s (batch_size=${bs:-n/a})"
            echo "cmd: ${CMD[*]}"
            echo "artifacts:"
            printf '%s' "$arts" | sed 's/^/  /'
        } > "$marker"
        say "${C_GRN}done${C_OFF} $id in ${elapsed}s"
        [[ -n "$arts" ]] && printf '%s\n' "$arts" | sed 's/^/      /'
        # record_artifacts runs in a subshell, so count from its output.
        local missing
        missing=$(printf '%s\n' "$arts" | grep -c '(MISSING)')
        (( missing > 0 )) && \
            warn "$id exited 0 but $missing expected artifact(s) are missing -- check $log"
        DONE_STEPS+=("$id")
        return 0
    fi

    warn "$id FAILED (exit $rc) after ${elapsed}s -- see $log"
    FAILED_STEPS+=("$id (exit $rc)")
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
        print_summary
        die "STOP_ON_FAIL=1 and $id failed"
    fi
    return "$rc"
}

# What ran, what did not, and where to look. Called from the normal end of the
# pipeline and from the interrupt path, so an aborted run still reports state.
print_summary() {
    local total=$(( $(date +%s) - PIPELINE_START ))
    rule
    if (( INTERRUPTED )); then
        say "pipeline INTERRUPTED after ${total}s${INTERRUPTED_STEP:+ during $INTERRUPTED_STEP}"
    else
        say "pipeline finished in ${total}s"
    fi
    say "state:     $MARK_DIR"
    say "logs:      $LOG_DIR"
    say "artifacts: $ART_DIR"
    say "manifest:  $MANIFEST"
    echo
    if (( ${#DONE_STEPS[@]} )); then
        echo "${C_GRN}completed (${#DONE_STEPS[@]}):${C_OFF}"
        printf '  %s\n' "${DONE_STEPS[@]}"
    fi
    if (( ${#SKIPPED_STEPS[@]} )); then
        echo "${C_DIM}skipped (${#SKIPPED_STEPS[@]}):${C_OFF}"
        printf '  %s\n' "${SKIPPED_STEPS[@]}"
    fi
    if (( ${#FAILED_STEPS[@]} )); then
        echo "${C_RED}failed (${#FAILED_STEPS[@]}):${C_OFF}"
        printf '  %s\n' "${FAILED_STEPS[@]}"
    fi
    echo

    # Last thing every exit path does, the interrupted one included: the logs of
    # a run that was killed are exactly the ones somebody wants to read later.
    publish_state

    echo
    say "re-run this script to continue: steps whose results are already complete"
    say "are skipped, and an interrupted or failed one resumes from its own"
    say "checkpoint at the epoch it reached."
}

# --------------------------------------------------------------------------- #
# step builders
# --------------------------------------------------------------------------- #

frac_tag() { echo "${1/./}"; }   # 0.25 -> 025
# The python side names its per-run files with "%g"-formatted fractions, so the
# driver has to normalise the same way to predict them: 0.50 -> 0.5.
frac_num() { printf '%g' "$1"; }

build_decoder() {
    local bs="$1"
    CMD=("${UV_RUN[@]}" lcm_scripts/blt_decoder.py
         --entropy_model "$ENTROPY_MODEL"
         --num_sentences "$DECODER_SENTENCES" --epochs "$DECODER_EPOCHS"
         --batch_size "$bs" --amp
         --pooler_save_path lcm_models/blt_pooler.pth
         --save_path lcm_models/blt_decoder.pth
         --resume auto)
    STEP_ARTIFACTS=(lcm_models/blt_decoder.pth lcm_models/blt_pooler.pth)
    # blt_decoder.py names its history and its published run after the save
    # path's basename, and writes both beside the checkpoints.
    STEP_PROBE=(--epochs "$DECODER_EPOCHS"
                --history lcm_models/blt_decoder_history.json
                --run-json "$RESULTS_RUNS_DIR/blt_decoder/run.json"
                --file lcm_models/blt_decoder.pth
                --file lcm_models/blt_pooler.pth)
    # Stage 2 loads blt_decoder.pth and blt_pooler.pth, and neither is ever
    # published, so someone else's finished decoder run does not spare this
    # machine the training unless the checkpoints are already here.
    STEP_KIND="producer"
    STEP_REMOTE_RUN="blt_decoder"
    STEP_REMOTE_SPEC=(--epochs "$DECODER_EPOCHS"
                      --expect "num_sentences=$DECODER_SENTENCES"
                      --expect "entropy_model=$ENTROPY_MODEL")
}

ENCODE_FRACTION=""
build_encode() {
    local bs="$1" f="$ENCODE_FRACTION" cache model_dir
    cache="embeddings/blt_embeddings_frac$(frac_tag "$f").pth"
    # One --model_dir per fraction. train_lcm_blt.py names its checkpoints
    # `lcm_blt_last/_best.pth` (the eval scripts glob for exactly that inside a
    # per-run directory), so the fractions have to be separated by directory:
    # sharing lcm_models/ put all three on the same file, and since the run
    # fingerprint covers --fraction, the second fraction aborted on the first
    # one's checkpoint instead of resuming its own.
    model_dir="lcm_models/blt_lcm_$(frac_tag "$f")"
    CMD=("${UV_RUN[@]}" lcm_scripts/train_lcm_blt.py
         --entropy_model "$ENTROPY_MODEL"
         --fraction "$f" --epochs 0 --batch_size "$bs"
         --model_dir "$model_dir"
         --embed_cache "$cache"
         --resume auto)
    # The cached embeddings feed the paper's numbers, so they stay fp32 unless
    # AMP_ENCODE=1 is set explicitly.
    [[ "${AMP_ENCODE:-0}" == "1" ]] && CMD+=(--amp)
    STEP_ARTIFACTS=("$cache" "$model_dir")
    # --epochs 0: this step encodes, it does not train, so the cache IS the
    # result. It is written whole (staged, then renamed) by cached_torch, so its
    # presence means the encode pass finished.
    STEP_PROBE=(--epochs 0
                --run-json "$RESULTS_RUNS_DIR/lcm_blt_base_fraction$(frac_num "$f")/run.json"
                --file "$cache")
    # Same as the decoder: the whole point of this step is the embedding cache,
    # which is far too large to publish. A published record means the encode
    # happened somewhere, not that this machine has the cache Stage 2 reads.
    STEP_KIND="producer"
    STEP_REMOTE_RUN="lcm_blt_base_fraction$(frac_num "$f")"
    STEP_REMOTE_SPEC=(--epochs 0
                      --expect "fraction=$f"
                      --expect "entropy_model=$ENTROPY_MODEL")
}

MT_FRACTION=""; MT_SEED=""
build_mt() {
    local bs="$1" f="$MT_FRACTION" s="$MT_SEED" csv run
    csv="results/blt_lcm_mt_${f}_s${s}.csv"
    run="lcm_blt_mt_fraction$(frac_num "$f")_s${s}"
    CMD=("${UV_RUN[@]}" lcm_scripts/train_lcm_blt_mt.py
         --entropy_model "$ENTROPY_MODEL"
         --pooler lcm_models/blt_pooler.pth
         --decoder lcm_models/blt_decoder.pth
         --fraction "$f" --epochs "$MT_EPOCHS" --batch_size "$bs"
         --seed "$s" --data_seed "$DATA_SEED"
         --noise_levels "${NOISE_LEVELS[@]}"
         --comet_model "$COMET_MODEL"
         --out_csv "$csv"
         --embed_cache "embeddings/blt_mt_concepts_frac$(frac_tag "$f").pth"
         --resume auto)
    # One concept cache per FRACTION, shared by all three seeds. The encoder is
    # frozen and runs under no_grad in eval mode, so the concepts of a fraction
    # are identical for every seed; train_lcm_blt_mt.py keys this cache on the
    # encoding inputs only, so seed 43 loads what seed 42 wrote. A per-seed
    # cache re-encoded the same corpus nine times across the grid: 4.2 GPU-hours
    # and 99 GiB to store three identical copies of each fraction.
    # The checkpoint carries the fraction as well as the seed -- the whole grid
    # writes into one --model_dir, and a seed-only name had every fraction
    # landing on the same file (which is what made this step fail).
    STEP_ARTIFACTS=("$csv" "lcm_models/${run}_best.pth")
    # A cell of the grid is only complete when the training ran its full
    # --epochs AND the evaluation produced a row for every noise level: the CSV
    # is written once, at the very end, after the last noise level is scored.
    STEP_PROBE=(--epochs "$MT_EPOCHS"
                --history "lcm_models/${run}_history.json"
                --run-json "$RESULTS_RUNS_DIR/${run}/run.json"
                --file "lcm_models/${run}_best.pth"
                --csv "$csv" --csv-rows "${#NOISE_LEVELS[@]}")
    # The headline grid. A cell exists to produce this CSV, and the CSV is
    # published alongside the record -- so a cell another machine has already
    # pushed is finished work, and nothing downstream needs its checkpoint.
    # The run name pins fraction and seed; --expect pins the rest of the cell.
    STEP_REMOTE_RUN="$run"
    STEP_REMOTE_CSV="$(basename "$csv")"
    STEP_REMOTE_SPEC=(--epochs "$MT_EPOCHS"
                      --csv-rows "${#NOISE_LEVELS[@]}"
                      --expect "fraction=$f"
                      --expect "seed=$s"
                      --expect "data_seed=$DATA_SEED"
                      --expect "noise_levels=${NOISE_LEVELS[*]}"
                      --expect "entropy_model=$ENTROPY_MODEL")
}

build_baselines() {
    CMD=("${UV_RUN[@]}" lcm_scripts/benchmark_bhashasetu_models.py
         --models "${BENCH_MODELS[@]}"
         --fractions "${FRACTIONS[@]}" --noise_levels "${NOISE_LEVELS[@]}"
         --eval_docs 100 --epochs "$BENCH_EPOCHS"
         --out_dir runs/bhashasetu_benchmarks)
    STEP_ARTIFACTS=(runs/bhashasetu_benchmarks)
    # The orchestrator concatenates one row per (model, fraction, noise) into
    # summary_metrics.csv after the last sub-job returns, so a full-length
    # summary is exactly the statement "every cell of the grid ran". Epoch
    # counting belongs to the sub-jobs, which resume themselves.
    STEP_PROBE=(--epochs 0
                --csv runs/bhashasetu_benchmarks/summary_metrics.csv
                --csv-rows "$(( ${#BENCH_MODELS[@]} * ${#FRACTIONS[@]} * ${#NOISE_LEVELS[@]} ))")
    # benchmark_bhashasetu_models.py publishes the summary CSV with its record,
    # so a full-length published summary is the whole deliverable of this step.
    STEP_REMOTE_RUN="bhashasetu_benchmark"
    STEP_REMOTE_CSV="summary_metrics.csv"
    STEP_REMOTE_SPEC=(--epochs 0
                      --csv-rows "$(( ${#BENCH_MODELS[@]} * ${#FRACTIONS[@]} * ${#NOISE_LEVELS[@]} ))"
                      --expect "epochs=$BENCH_EPOCHS"
                      --expect "noise_levels=${NOISE_LEVELS[*]}")
}

build_llama() {
    CMD=("${UV_RUN[@]}" lcm_scripts/train_bpe_llama8b.py
         --fraction 0.25 --epochs "$LLAMA_EPOCHS" --batch_size 1 --grad_accum 16
         --qlora --noise_levels "${NOISE_LEVELS[@]}"
         --out_dir runs/bpe_llama8b_25_qlora
         --resume auto)
    STEP_ARTIFACTS=(runs/bpe_llama8b_25_qlora)
    STEP_PROBE=(--epochs "$LLAMA_EPOCHS"
                --history runs/bpe_llama8b_25_qlora/bpe_llama8b_fraction0.25_history.json
                --run-json "$RESULTS_RUNS_DIR/bpe_llama8b_fraction0.25/run.json"
                --csv runs/bpe_llama8b_25_qlora/metrics_fraction0.25.csv
                --csv-rows "${#NOISE_LEVELS[@]}")
    STEP_REMOTE_RUN="bpe_llama8b_fraction0.25"
    STEP_REMOTE_CSV="metrics_fraction0.25.csv"
    STEP_REMOTE_SPEC=(--epochs "$LLAMA_EPOCHS"
                      --csv-rows "${#NOISE_LEVELS[@]}"
                      --expect "fraction=0.25"
                      --expect "noise_levels=${NOISE_LEVELS[*]}")
}

ANALYSIS_SCRIPT=""
build_analysis() {
    CMD=("${UV_RUN[@]}" "$ANALYSIS_SCRIPT")
    STEP_ARTIFACTS=()
    # The corpus analyses have no epochs and no single result file to key on;
    # each one resumes per record through its own ResumableJsonl output, so
    # re-running a finished analysis is cheap and there is nothing to adopt.
    # They do not publish a run record either, so there is nothing to read from
    # the results refs for them.
    STEP_PROBE=()
}

# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #

# PIPELINE_START is set with the signal trap, well before this point, so the
# interrupt path can report an elapsed time no matter where it fires.

if (( LIST_ONLY )); then
    say "planned steps (VRAM ${VRAM} MiB):"
fi

# --- install ---------------------------------------------------------------- #
INSTALL_MARK="$MARK_DIR/setup_deps.done"
if (( PUBLISH_STATE_ONLY )); then
    # Publishing a state directory does not need the venv the experiments run in.
    SKIPPED_STEPS+=("setup:deps (--publish-state)")
elif selected "setup:deps"; then
    if [[ -f "$INSTALL_MARK" && $FORCE -eq 0 ]]; then
        info "skip ${C_BOLD}setup:deps${C_OFF} -- already done"
    elif (( LIST_ONLY )); then
        printf '  %-28s %s\n' "setup:deps" "uv pip install torch/torchvision (cu130) && uv sync"
    else
        rule
        say "${C_BOLD}setup:deps${C_OFF} -- installing dependencies"
        if install_deps 2>&1 | tee "$LOG_DIR/setup_deps.log"; then
            (( DRY_RUN )) || echo "finished $(now)" > "$INSTALL_MARK"
            DONE_STEPS+=("setup:deps")
        elif (( INTERRUPTED )); then
            INTERRUPTED_STEP="setup:deps"
            print_summary
            exit 130
        else
            die "dependency install failed -- see $LOG_DIR/setup_deps.log"
        fi
    fi
else
    SKIPPED_STEPS+=("setup:deps (filtered)")
fi

# --- decoder + pooler ------------------------------------------------------- #
run_step "decoder" \
         "BLT byte decoder + pooler ($DECODER_SENTENCES sentences, $DECODER_EPOCHS epochs)" \
         build_decoder "$BS_DECODER"

# --- Stage 1: pre-encode BLT embeddings ------------------------------------- #
for f in "${FRACTIONS[@]}"; do
    ENCODE_FRACTION="$f"
    run_step "encode:$f" "Stage 1 -- pre-encode BLT embeddings (fraction $f)" \
             build_encode "$BS_ENCODE"
done

# --- Stage 2: EN->MR translation grid (headline result) --------------------- #
if [[ ! -f lcm_models/blt_decoder.pth || ! -f lcm_models/blt_pooler.pth ]] \
   && (( ! DRY_RUN )) && (( ! LIST_ONLY )) && (( ! PUBLISH_STATE_ONLY )); then
    warn "lcm_models/blt_decoder.pth or blt_pooler.pth missing -- Stage 2 needs both."
    warn "Skipping the MT grid; fix the 'decoder' step and re-run."
    SKIPPED_STEPS+=("mt:* (decoder/pooler missing)")
else
    for s in "${SEEDS[@]}"; do
        for f in "${FRACTIONS[@]}"; do
            MT_FRACTION="$f"; MT_SEED="$s"
            run_step "mt:f${f}_s${s}" \
                     "Stage 2 -- EN->MR MT (fraction $f, seed $s, $MT_EPOCHS epochs, noise ${NOISE_LEVELS[*]})" \
                     build_mt "$BS_MT"
        done
    done
fi

# --- Stage 3: baselines ----------------------------------------------------- #
run_step "bench:baselines" \
         "Stage 3 -- BPE-Transformer + BPE-LCM + SONAR-LCM benchmark grid" \
         build_baselines

if [[ "${RUN_LLAMA:-0}" == "1" ]]; then
    if (( VRAM >= 20000 )) || (( DRY_RUN )) || (( LIST_ONLY )); then
        run_step "bench:llama8b" "Stage 3 -- Llama-8B QLoRA baseline (fraction 0.25)" \
                 build_llama
    else
        warn "RUN_LLAMA=1 but only ${VRAM} MiB VRAM -- Llama-8B QLoRA needs ~20 GiB. Skipping."
        SKIPPED_STEPS+=("bench:llama8b (insufficient VRAM)")
    fi
else
    info "skip bench:llama8b -- set RUN_LLAMA=1 to include it (needs HF auth + ~20 GiB)"
    SKIPPED_STEPS+=("bench:llama8b (not requested)")
fi

# --- Stage 4: linguistic analyses (CPU) ------------------------------------- #
ANALYSES=(
    "analysis:fertility_chrf|lcm_scripts/fertility_chrf_scatter.py"
    "analysis:fertility_audit|lcm_scripts/fertility_audit.py"
    "analysis:morpheme_align|morpheme_alignment/morpheme_boundary_alignment.py"
    "analysis:morpheme_fig2|morpheme_alignment/patch_morpheme_example.py"
    "analysis:chunk_ablation|fixed_chunk_ablation/fixed_chunk_ablation.py"
    "analysis:threshold_sweep|sweep_threshold/sweep_entropy_threshold.py"
    "analysis:patch_compression|tokenization_statistics/patch_compression_by_morpheme_class.py"
    "analysis:hindi_sanity|cross_script_sanity/hindi_entropy_sanity.py"
)
for entry in "${ANALYSES[@]}"; do
    id="${entry%%|*}"; script="${entry##*|}"
    if [[ ! -f "$script" ]] && (( ! DRY_RUN )); then
        warn "$id -- $script not found in this checkout, skipping"
        SKIPPED_STEPS+=("$id (script missing)")
        continue
    fi
    ANALYSIS_SCRIPT="$script"
    run_step "$id" "Stage 4 -- $(basename "$script")" build_analysis
done

# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

if (( LIST_ONLY )); then
    rule
    say "listing only -- nothing was run"
    exit 0
fi

print_summary
if (( ${#FAILED_STEPS[@]} )); then
    exit 1
fi
if (( PUBLISH_STATE_ONLY )); then
    say "--publish-state finished: nothing was run, and the state directory is published."
else
    say "all selected steps completed."
fi
