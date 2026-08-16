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
# Ctrl-C stops the pipeline, not just the running step: the interrupted step is
# left unmarked (so a re-run resumes it) and nothing further is started.
#
# Usage (from anywhere -- the script finds its own repo root):
#   bash scripts/auto_setup.sh                  # everything, resuming as needed
#   bash scripts/auto_setup.sh --list           # show the plan + chosen batch sizes
#   bash scripts/auto_setup.sh --dry-run        # print commands, run nothing
#   bash scripts/auto_setup.sh --only 'mt:*'    # only the Stage 2 MT grid
#   bash scripts/auto_setup.sh --skip 'bench:*' # everything except the baselines
#   bash scripts/auto_setup.sh --force --only decoder
#
# Environment knobs (all optional):
#   REPO_DIR      repository root  (default: the parent of this script)
#   STATE_DIR     progress + artifacts root   (default: $REPO_DIR/runs/auto_setup)
#   VRAM_MB       override the probed VRAM in MiB (e.g. for a shared GPU)
#   BS_DECODER / BS_ENCODE / BS_MT   override a stage's batch size outright
#   MT_ALLOW_LARGER_BATCH=1   let Stage 2 exceed the script default of 32
#   AMP_ENCODE=1  add --amp to the Stage 1 encode (off: cached embeddings stay fp32)
#   RUN_LLAMA=1   also run the Llama-8B QLoRA baseline (needs >=20 GiB + HF auth)
#   KEEP_CU130=1  pass --no-sync to `uv run` so the cu130 wheels survive (see NOTE)
#   UV_RUN_OVERRIDE  replace the `uv run` launcher entirely (apptainer, srun, tests)
#   STOP_ON_FAIL=1  abort on the first failing experiment step
#   MAX_OOM_RETRIES  halve the batch size and retry this many times (default 3)
#
# NOTE on cu130: pyproject.toml pins `torch==2.5.1` from the cu121 index, and
# `uv run` re-syncs the venv on every invocation. The requested
# `uv pip install ... /cu130 && uv sync` therefore installs cu130 wheels that
# `uv sync` -- and then every later `uv run` -- reverts to cu121 torch 2.5.1.
# The script performs the requested install verbatim, reports the torch build
# that actually ends up in the venv before and after the sync, and offers
# KEEP_CU130=1 (adds `--no-sync` to `uv run`) if you want the cu130 install to
# stick. The durable fix is to bump the pin/index in pyproject.toml.

set -uo pipefail

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

ENTROPY_MODEL=${ENTROPY_MODEL:-patching_scratch/entropy_model_marathi.pt}
COMET_MODEL=${COMET_MODEL:-Unbabel/wmt22-comet-da}
FRACTIONS=(0.25 0.50 0.80)
SEEDS=(42 43 44)
DATA_SEED=${DATA_SEED:-42}
MAX_OOM_RETRIES=${MAX_OOM_RETRIES:-3}
STOP_ON_FAIL=${STOP_ON_FAIL:-0}

ONLY_PAT=""
SKIP_PAT=""
FORCE=0
DRY_RUN=0
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)    ONLY_PAT="$2"; shift 2 ;;
        --skip)    SKIP_PAT="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --list)    LIST_ONLY=1; shift ;;
        -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
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
        warn "uv sync replaced the cu130 torch with the pyproject pin ($after)."
        warn "Every 'uv run' re-syncs too. Set KEEP_CU130=1 to run with --no-sync,"
        warn "or bump the torch pin / index in pyproject.toml for a durable fix."
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
say "GPU allocation:"
gpu_report
if (( VRAM == 0 )); then
    warn "no CUDA GPU visible -- everything below will run on CPU and be far too"
    warn "slow for the training stages. Only Stage 4 is genuinely CPU-friendly."
else
    say "usable VRAM: ${VRAM} MiB (smallest visible device)"
fi
say "batch sizes: decoder=${BS_DECODER}  encode=${BS_ENCODE}  mt=${BS_MT}"
rule

if (( ! DRY_RUN )); then
    {
        printf '{"timestamp":"%s","vram_mb":%s,"bs_decoder":%s,"bs_encode":%s,"bs_mt":%s}\n' \
            "$(now)" "$VRAM" "$BS_DECODER" "$BS_ENCODE" "$BS_MT"
    } > "$ART_DIR/gpu_probe.json"
    gpu_report > "$ART_DIR/gpu_report.txt" 2>&1
fi

# --------------------------------------------------------------------------- #
# step runner: logging, resume markers, artifact capture, OOM backoff
# --------------------------------------------------------------------------- #

CMD=()               # set by each step's builder function
STEP_ARTIFACTS=()    # paths a step is expected to produce
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
    if [[ -f "$marker" && $FORCE -eq 0 ]]; then
        info "skip ${C_BOLD}$id${C_OFF} -- already done ($(head -1 "$marker" 2>/dev/null))"
        SKIPPED_STEPS+=("$id (done)")
        return 0
    fi

    "$builder" "$bs"

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
    say "re-run this script to continue: completed steps are skipped, and an"
    say "interrupted or failed one resumes from its own checkpoint."
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
         --num_sentences 50000 --epochs 10 --batch_size "$bs" --amp
         --pooler_save_path lcm_models/blt_pooler.pth
         --save_path lcm_models/blt_decoder.pth
         --resume auto)
    STEP_ARTIFACTS=(lcm_models/blt_decoder.pth lcm_models/blt_pooler.pth)
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
}

MT_FRACTION=""; MT_SEED=""
build_mt() {
    local bs="$1" f="$MT_FRACTION" s="$MT_SEED" csv
    csv="results/blt_lcm_mt_${f}_s${s}.csv"
    CMD=("${UV_RUN[@]}" lcm_scripts/train_lcm_blt_mt.py
         --entropy_model "$ENTROPY_MODEL"
         --pooler lcm_models/blt_pooler.pth
         --decoder lcm_models/blt_decoder.pth
         --fraction "$f" --epochs 3 --batch_size "$bs"
         --seed "$s" --data_seed "$DATA_SEED"
         --noise_levels 0.0 0.1 0.2
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
    STEP_ARTIFACTS=(
        "$csv"
        "lcm_models/lcm_blt_mt_fraction$(frac_num "$f")_s${s}_best.pth"
    )
}

build_baselines() {
    CMD=("${UV_RUN[@]}" lcm_scripts/benchmark_bhashasetu_models.py
         --models bpe_transformer bpe_lcm sonar_lcm
         --fractions 0.25 0.50 0.80 --noise_levels 0.0 0.1 0.2
         --eval_docs 100 --epochs 1
         --out_dir runs/bhashasetu_benchmarks)
    STEP_ARTIFACTS=(runs/bhashasetu_benchmarks)
}

build_llama() {
    CMD=("${UV_RUN[@]}" lcm_scripts/train_bpe_llama8b.py
         --fraction 0.25 --epochs 1 --batch_size 1 --grad_accum 16
         --qlora --noise_levels 0.0 0.1 0.2
         --out_dir runs/bpe_llama8b_25_qlora
         --resume auto)
    STEP_ARTIFACTS=(runs/bpe_llama8b_25_qlora)
}

ANALYSIS_SCRIPT=""
build_analysis() {
    CMD=("${UV_RUN[@]}" "$ANALYSIS_SCRIPT")
    STEP_ARTIFACTS=()
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
if selected "setup:deps"; then
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
run_step "decoder" "BLT byte decoder + pooler (50k sentences, 10 epochs)" \
         build_decoder "$BS_DECODER"

# --- Stage 1: pre-encode BLT embeddings ------------------------------------- #
for f in "${FRACTIONS[@]}"; do
    ENCODE_FRACTION="$f"
    run_step "encode:$f" "Stage 1 -- pre-encode BLT embeddings (fraction $f)" \
             build_encode "$BS_ENCODE"
done

# --- Stage 2: EN->MR translation grid (headline result) --------------------- #
if [[ ! -f lcm_models/blt_decoder.pth || ! -f lcm_models/blt_pooler.pth ]] \
   && (( ! DRY_RUN )) && (( ! LIST_ONLY )); then
    warn "lcm_models/blt_decoder.pth or blt_pooler.pth missing -- Stage 2 needs both."
    warn "Skipping the MT grid; fix the 'decoder' step and re-run."
    SKIPPED_STEPS+=("mt:* (decoder/pooler missing)")
else
    for s in "${SEEDS[@]}"; do
        for f in "${FRACTIONS[@]}"; do
            MT_FRACTION="$f"; MT_SEED="$s"
            run_step "mt:f${f}_s${s}" \
                     "Stage 2 -- EN->MR MT (fraction $f, seed $s, noise 0/0.1/0.2)" \
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
say "all selected steps completed."
