"""Publish a run's results to the git repository, and optionally push them.

Every training and evaluation entry point ends by collecting what the run
produced -- the figures, the metric CSV/JSON, the loss history, and the full
hyperparameter set including the resolved device, git commit and library
versions -- into ``results/runs/<run_name>/`` and committing it. With
``--push_results`` it also pushes, so a cluster job's outputs land on GitHub
without anyone copying files off the node.

Deliberate limits, because this runs unattended at the end of every job:

* **Only an explicit file list is staged.** ``git add <paths>``, never ``-A``.
  A run cannot commit your working-tree edits, and cannot commit a checkpoint,
  a dataset shard or an embedding cache.
* **Size-capped.** Anything over ``--results_max_mb`` is skipped with a warning
  rather than pushed into the repository's history, where it is permanent.
* **Never fatal.** No remote, no upstream, a rejected push, a dirty index, a
  detached HEAD -- all print a warning and return. A finished training run is
  never lost to a failed push.
* **Never interactive.** Every git call runs with terminal prompts disabled and
  a wall-clock timeout, so a missing credential fails in seconds instead of
  hanging a GPU job until its Slurm time limit.
* **Push is opt-in.** Committing locally is cheap and reversible; pushing is
  neither, so it takes ``--push_results`` (or ``BLT_LCM_PUSH_RESULTS=1``).

Cluster mode
------------
On a scheduler (Slurm/PBS/LSF, auto-detected) the commit is built through a
*temporary index* and pushed straight to the remote, without touching the
shared checkout's index, HEAD or working tree. This matters because array jobs
all ``cd $REPO_DIR`` into one clone: concurrent ``git add`` would fight over
``.git/index.lock``, and a job that moved HEAD would do it underneath every
other job still running. See ``_publish_isolated``.
"""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Optional, Sequence

# Output/destination switches only; see checkpoint_utils.DEFAULT_FINGERPRINT_IGNORE.
RESULTS_ARG_NAMES = (
    "results_dir",
    "push_results",
    "no_results",
    "results_remote",
    "results_branch",
    "results_max_mb",
    "results_commit_mode",
    "results_retries",
    "results_timeout",
)

# Extensions worth publishing. Checkpoints (.pth/.pt), caches (.pth), corpora
# (.jsonl) and tokenizer models are deliberately absent -- they are large,
# regenerable, and belong in artifact storage rather than in git history.
PUBLISHABLE_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf", ".svg", ".csv", ".json", ".md", ".txt")

# Files that are outputs of the run but not results anybody wants in the repo.
SKIP_NAMES = ("bpe_corpus.txt", "bpe_lcm_corpus.txt")

# Environment variables a scheduler sets. Their presence is what switches the
# publisher into the shared-checkout-safe path.
SCHEDULER_ENV_VARS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "PBS_JOBID",
    "LSB_JOBID",
    "SGE_TASK_ID",
)

# Consulted, in order, for a token to authenticate an https remote.
TOKEN_ENV_VARS = ("BLT_LCM_GIT_TOKEN", "GIT_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")

DEFAULT_GIT_TIMEOUT = 120.0


def on_scheduler() -> Optional[str]:
    """The scheduler job id, if this process is running under one."""
    for var in SCHEDULER_ENV_VARS:
        if os.environ.get(var):
            return f"{var}={os.environ[var]}"
    return None


def add_results_args(parser, *, default_dir: str = "results/runs") -> None:
    """Add the results-publishing flags to a script's parser."""
    parser.add_argument(
        "--results_dir",
        type=str,
        default=default_dir,
        help="Repository directory that collected run results are written to.",
    )
    parser.add_argument(
        "--push_results",
        action="store_true",
        default=os.environ.get("BLT_LCM_PUSH_RESULTS", "") == "1",
        help="After committing the collected results, push them to the remote. "
        "Also enabled by BLT_LCM_PUSH_RESULTS=1, which is the practical way to "
        "turn it on for every job in a Slurm submission script.",
    )
    parser.add_argument(
        "--no_results",
        action="store_true",
        help="Do not collect or commit results for this run.",
    )
    parser.add_argument(
        "--results_remote",
        type=str,
        default=os.environ.get("BLT_LCM_RESULTS_REMOTE", "origin"),
        help="Remote to push collected results to.",
    )
    parser.add_argument(
        "--results_branch",
        type=str,
        default=os.environ.get("BLT_LCM_RESULTS_BRANCH") or None,
        help="Branch to push results to. Defaults to the current branch.",
    )
    parser.add_argument(
        "--results_max_mb",
        type=float,
        default=25.0,
        help="Skip any single result file larger than this, rather than "
        "committing it into git history permanently.",
    )
    parser.add_argument(
        "--results_commit_mode",
        choices=["auto", "isolated", "worktree"],
        default=os.environ.get("BLT_LCM_RESULTS_COMMIT_MODE", "auto"),
        help="'isolated' builds the commit through a temporary index and pushes "
        "it without touching the checkout's index/HEAD -- required when several "
        "cluster jobs share one clone. 'worktree' is the ordinary "
        "add/commit/push. 'auto' (default) picks isolated under a scheduler.",
    )
    parser.add_argument(
        "--results_retries",
        type=int,
        default=int(os.environ.get("BLT_LCM_RESULTS_RETRIES", "5")),
        help="Attempts for a push that loses the race to a concurrent job. "
        "Backoff is randomized, so an array of jobs finishing together does "
        "not retry in lockstep.",
    )
    parser.add_argument(
        "--results_timeout",
        type=float,
        default=float(os.environ.get("BLT_LCM_RESULTS_TIMEOUT", DEFAULT_GIT_TIMEOUT)),
        help="Per-git-command timeout in seconds. Bounds how long a hung "
        "network call can hold up the end of a job.",
    )


# --------------------------------------------------------------------------- #
# git helpers -- every one of these is allowed to fail, and none may block
# --------------------------------------------------------------------------- #


def _git_env() -> dict:
    """Environment for every git call: non-interactive, no prompts, ever.

    A compute node has no terminal. Without these, a missing or expired
    credential makes git block on a username prompt and the job sits idle on a
    GPU until Slurm kills it at the wall-clock limit -- the single worst failure
    mode for unattended publishing.
    """
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_ASKPASS", "echo")
    env.setdefault("SSH_ASKPASS", "echo")
    env["GIT_PAGER"] = "cat"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def _identity_args(root: str) -> list[str]:
    """``-c`` overrides so a commit works on a node with no git identity.

    Compute nodes routinely have no ``user.email``/``user.name``, and git then
    refuses to commit at all. A configured identity always wins; this only
    fills the gap.
    """
    args = [
        # Shared filesystems hand the checkout to a different uid than the one
        # that cloned it, and git then refuses to operate on it at all.
        "-c", f"safe.directory={root}",
    ]
    have = _git_raw("config", "user.email", cwd=root, check_identity=False)
    if have.returncode != 0 or not have.stdout.strip():
        args += [
            "-c", f"user.email={os.environ.get('GIT_AUTHOR_EMAIL', 'blt-lcm-runs@localhost')}",
            "-c", f"user.name={os.environ.get('GIT_AUTHOR_NAME', 'BLT-LCM run publisher')}",
        ]
    return args


def _git_raw(
    *args: str,
    cwd: str,
    timeout: float = DEFAULT_GIT_TIMEOUT,
    extra_env: Optional[dict] = None,
    check_identity: bool = True,
) -> subprocess.CompletedProcess:
    env = _git_env()
    if extra_env:
        env.update(extra_env)
    cmd = ["git", *args]
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd, 124, "", f"git timed out after {timeout:g}s"
        )
    except Exception as e:  # git missing entirely
        return subprocess.CompletedProcess(cmd, 127, "", str(e))


def _repo_root(start: str) -> Optional[str]:
    r = _git_raw("rev-parse", "--show-toplevel", cwd=start)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _current_branch(root: str) -> Optional[str]:
    r = _git_raw("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    branch = r.stdout.strip() if r.returncode == 0 else ""
    # "HEAD" means a detached checkout; there is nothing to push to.
    return branch if branch and branch != "HEAD" else None


def _head_commit(root: str) -> Optional[str]:
    r = _git_raw("rev-parse", "--short", "HEAD", cwd=root)
    return r.stdout.strip() if r.returncode == 0 else None


def _authenticated_url(url: str) -> Optional[str]:
    """An https remote URL carrying a token from the environment, if there is one.

    Cluster nodes rarely have a credential helper or an ssh agent, and a
    compute node has no terminal to prompt on, so a token in the job
    environment is the practical way in. Put it in ``.env`` (gitignored), which
    the submission scripts source; never in a tracked file, because the first
    results push would publish it.

    The assembled URL is never printed -- ``_push`` redacts it out of any error
    it reports.
    """
    if not url.startswith("https://"):
        return None  # ssh remotes authenticate with the agent/key
    token = next((os.environ[v] for v in TOKEN_ENV_VARS if os.environ.get(v)), None)
    if not token:
        return None
    rest = url[len("https://") :]
    if "@" in rest.split("/", 1)[0]:  # already carries credentials
        return None
    # GitHub accepts any username alongside a PAT; `x-access-token` is the
    # documented placeholder and works for App tokens too. A configured
    # username is used when present, for hosts that are stricter.
    user = os.environ.get("GITHUB_USERNAME") or os.environ.get("GIT_USERNAME")
    return f"https://{user or 'x-access-token'}:{token}@{rest}"


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #


class ResultsRecorder:
    """Collects one run's artifacts under ``results/runs/<run_name>/``."""

    def __init__(
        self,
        args: Any,
        run_name: str,
        *,
        script: Optional[str] = None,
        fingerprint: Optional[str] = None,
        repo_root: Optional[str] = None,
    ):
        self.run_name = run_name
        self.script = script or os.path.basename(sys.argv[0] or "unknown")
        self.fingerprint = fingerprint
        self.args = args
        self.enabled = not getattr(args, "no_results", False)
        self.push = bool(getattr(args, "push_results", False))
        self.remote = getattr(args, "results_remote", "origin")
        self.branch = getattr(args, "results_branch", None)
        self.max_bytes = float(getattr(args, "results_max_mb", 25.0)) * 1024 * 1024
        self.retries = max(int(getattr(args, "results_retries", 5)), 1)
        self.timeout = float(getattr(args, "results_timeout", DEFAULT_GIT_TIMEOUT))
        self.started = time.time()
        self.metrics: dict[str, Any] = {}
        self.extra: dict[str, Any] = {}
        self._sources: list[str] = []

        self.repo_root = repo_root or _repo_root(os.getcwd())
        if self.enabled and not self.repo_root:
            print("[results] not inside a git repository; results will not be committed")
            self.enabled = False

        mode = getattr(args, "results_commit_mode", "auto")
        job = on_scheduler()
        self.isolated = mode == "isolated" or (mode == "auto" and job is not None)
        if self.enabled and self.isolated and job:
            print(
                f"[results] scheduler detected ({job}); using an isolated commit "
                "so parallel jobs sharing this checkout do not race on the index"
            )

        rel = getattr(args, "results_dir", "results/runs") or "results/runs"
        self.out_dir = (
            os.path.join(self.repo_root, rel, run_name) if self.repo_root else None
        )

    # -- small git wrapper bound to this run's settings ---------------------- #

    def _git(self, *args: str, extra_env: Optional[dict] = None):
        assert self.repo_root
        return _git_raw(
            *args, cwd=self.repo_root, timeout=self.timeout, extra_env=extra_env
        )

    def _git_id(self, *args: str, extra_env: Optional[dict] = None):
        """git with the identity/safe.directory fallbacks applied."""
        assert self.repo_root
        return _git_raw(
            *_identity_args(self.repo_root),
            *args,
            cwd=self.repo_root,
            timeout=self.timeout,
            extra_env=extra_env,
        )

    # -- collection --------------------------------------------------------- #

    def add_source(self, *paths: str) -> None:
        """Register a file, or a directory to scan, as a source of results."""
        for p in paths:
            if p:
                self._sources.append(p)

    def add_metrics(self, **values: Any) -> None:
        """Record scalar results (final losses, BLEU/chrF++/TER, timings)."""
        self.metrics.update(values)

    def add_info(self, **values: Any) -> None:
        """Record anything else worth keeping (stop reason, dataset sizes)."""
        self.extra.update(values)

    # -- the run record ----------------------------------------------------- #

    def _environment(self) -> dict:
        env: dict[str, Any] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
        }
        if self.repo_root:
            env["git_commit"] = _head_commit(self.repo_root)
            env["git_branch"] = _current_branch(self.repo_root)
            # A dirty tree means the committed results were produced by code
            # that is not what the recorded commit contains. Say so.
            status = self._git("status", "--porcelain")
            env["git_dirty"] = bool(status.stdout.strip())
        try:
            import torch

            env["torch"] = torch.__version__
            env["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                env["cuda"] = torch.version.cuda
                env["gpu"] = torch.cuda.get_device_name(0)
                env["gpu_count"] = torch.cuda.device_count()
        except Exception:
            pass
        # Scheduler context, so a result can be traced back to its job log.
        for var in (
            "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION",
            "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "SLURM_CPUS_PER_TASK",
            "PBS_JOBID", "LSB_JOBID",
        ):
            if os.environ.get(var):
                env[var] = os.environ[var]
        return env

    def _record(self) -> dict:
        hyper = {}
        for k, v in sorted(vars(self.args).items() if hasattr(self.args, "__dict__") else []):
            try:
                json.dumps(v)
                hyper[k] = v
            except TypeError:
                hyper[k] = str(v)
        return {
            "run_name": self.run_name,
            "script": self.script,
            "fingerprint": self.fingerprint,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_clock_seconds": round(time.time() - self.started, 1),
            "hyperparameters": hyper,
            "metrics": self.metrics,
            "info": self.extra,
            "environment": self._environment(),
        }

    def _write_summary(self, record: dict, copied: Sequence[str]) -> str:
        assert self.out_dir
        path = os.path.join(self.out_dir, "README.md")
        lines = [
            f"# {self.run_name}",
            "",
            f"* script: `{record['script']}`",
            f"* finished: {record['finished_utc']}",
            f"* wall clock: {record['wall_clock_seconds']} s",
        ]
        env = record["environment"]
        if env.get("git_commit"):
            dirty = " (working tree dirty)" if env.get("git_dirty") else ""
            lines.append(f"* code: `{env['git_commit']}` on `{env.get('git_branch')}`{dirty}")
        if env.get("gpu"):
            lines.append(f"* device: {env['gpu']}")
        if env.get("SLURM_JOB_ID"):
            lines.append(
                f"* slurm job: `{env['SLURM_JOB_ID']}` on `{env.get('SLURM_JOB_NODELIST', '?')}`"
            )
        if record["metrics"]:
            lines += ["", "## Results", "", "| metric | value |", "| --- | --- |"]
            for k, v in record["metrics"].items():
                lines.append(f"| {k} | {v} |")
        if record["info"]:
            lines += ["", "## Run info", "", "| key | value |", "| --- | --- |"]
            for k, v in record["info"].items():
                lines.append(f"| {k} | {v} |")
        figures = [f for f in copied if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if figures:
            lines += ["", "## Figures", ""]
            for f in sorted(figures):
                rel = os.path.relpath(f, self.out_dir).replace(os.sep, "/")
                lines.append(f"### {os.path.basename(f)}")
                lines.append("")
                lines.append(f"![{os.path.basename(f)}]({rel})")
                lines.append("")
        lines += [
            "",
            "Full hyperparameters, metrics and environment: "
            "[`run.json`](run.json).",
            "",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    # -- publication -------------------------------------------------------- #

    def _iter_candidates(self) -> Iterable[str]:
        seen = set()
        for src in self._sources:
            if os.path.isdir(src):
                for name in sorted(os.listdir(src)):
                    p = os.path.join(src, name)
                    if os.path.isfile(p):
                        yield from self._accept(p, seen)
            elif os.path.isfile(src):
                yield from self._accept(src, seen)

    def _accept(self, path: str, seen: set) -> Iterable[str]:
        real = os.path.realpath(path)
        if real in seen:
            return
        name = os.path.basename(path)
        if name in SKIP_NAMES or not name.lower().endswith(PUBLISHABLE_SUFFIXES):
            return
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        if size > self.max_bytes:
            print(
                f"[results] skipping {name} ({size / 1024 / 1024:.1f} MB > "
                f"--results_max_mb); git history is permanent"
            )
            return
        seen.add(real)
        yield path

    def publish(self, *, message: Optional[str] = None) -> Optional[str]:
        """Collect, write the record, commit, and optionally push.

        Returns the results directory, or None when nothing was published.
        Never raises: a publishing failure must not fail a finished run.
        """
        if not self.enabled or not self.out_dir:
            return None
        try:
            return self._publish(message)
        except Exception as e:  # pragma: no cover - defensive by design
            print(f"[results] could not publish results: {e}")
            return None

    def _collect(self) -> tuple[list[str], dict]:
        assert self.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        copied = []
        for src in self._iter_candidates():
            dst = os.path.join(self.out_dir, os.path.basename(src))
            if os.path.realpath(src) == os.path.realpath(dst):
                copied.append(dst)
                continue
            shutil.copy2(src, dst)
            copied.append(dst)

        record = self._record()
        record_path = os.path.join(self.out_dir, "run.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        summary_path = self._write_summary(record, copied)
        return sorted(set(copied + [record_path, summary_path])), record

    def _publish(self, message: Optional[str]) -> Optional[str]:
        if not self.out_dir:
            return None
        paths, _record = self._collect()
        print(f"[results] collected {len(paths)} file(s) into {self.out_dir}")

        if not self.repo_root:
            return self.out_dir

        rel_paths = [
            os.path.relpath(p, self.repo_root).replace(os.sep, "/") for p in paths
        ]
        msg = message or (
            f"results: {self.run_name} ({self.script})\n\n"
            + "\n".join(f"{k}: {v}" for k, v in list(self.metrics.items())[:12])
        )

        if self.isolated:
            self._publish_isolated(rel_paths, msg)
        else:
            self._publish_worktree(rel_paths, msg)
        return self.out_dir

    # -- ordinary path: stage, commit, push --------------------------------- #

    def _publish_worktree(self, rel_paths: list[str], msg: str) -> None:
        add = self._git_id("add", "--", *rel_paths)
        if add.returncode != 0:
            # .gitignore is the usual cause; -f would override a deliberate
            # decision by whoever wrote that ignore rule, so it is not used.
            print(f"[results] git add failed: {add.stderr.strip()}")
            return

        staged = self._git("diff", "--cached", "--name-only", "--", *rel_paths)
        if not staged.stdout.strip():
            print("[results] results unchanged since the last run; nothing to commit")
            return

        # Only the explicitly staged result paths are committed. Anything else
        # the user happens to have staged is left alone.
        commit = self._git_id("commit", "-m", msg, "--only", "--", *rel_paths)
        if commit.returncode != 0:
            print(
                f"[results] git commit failed: "
                f"{commit.stderr.strip() or commit.stdout.strip()}"
            )
            return
        print(f"[results] committed {len(rel_paths)} file(s) for {self.run_name}")

        if not self._should_push():
            return
        branch = self._target_branch()
        if not branch:
            return
        for attempt in range(1, self.retries + 1):
            push = self._push(f"HEAD:{branch}", branch)
            if push is True:
                return
            if attempt == self.retries:
                break
            self._backoff(attempt)
            pull = self._git_id("pull", "--rebase", self.remote, branch)
            if pull.returncode != 0:
                print(f"[results] rebase failed: {pull.stderr.strip()}")
                self._git("rebase", "--abort")
                return
        print("[results] giving up on push; the commit is still here locally")

    # -- cluster path: temporary index, no working-tree mutation ------------ #

    def _publish_isolated(self, rel_paths: list[str], msg: str) -> None:
        """Build and push a commit without touching the shared checkout.

        Several array jobs run out of one clone. ``git add`` there takes
        ``.git/index.lock``, so concurrent jobs either fail or serialize badly,
        and a job that moved HEAD would move it under every other job still
        running. Instead the commit is assembled in a private index file with
        plumbing (read-tree / add / write-tree / commit-tree) and pushed
        directly, so the only shared state touched is the remote ref.
        """
        if not self._should_push():
            # Without a push there is nowhere for an isolated commit to go, so
            # fall back to the ordinary path and let it land locally.
            print(
                "[results] isolated mode needs --push_results to be useful; "
                "committing into the checkout instead"
            )
            self._publish_worktree(rel_paths, msg)
            return

        branch = self._target_branch()
        if not branch:
            return

        with tempfile.TemporaryDirectory(prefix="blt-lcm-results-") as tmp:
            index_file = os.path.join(tmp, "index")
            for attempt in range(1, self.retries + 1):
                env = {"GIT_INDEX_FILE": index_file}
                if os.path.exists(index_file):
                    os.remove(index_file)

                base = self._remote_tip(branch) or self._git(
                    "rev-parse", "HEAD"
                ).stdout.strip()
                if not base:
                    print("[results] no base commit to build on; skipping push")
                    return

                read = self._git("read-tree", base, extra_env=env)
                if read.returncode != 0:
                    print(f"[results] read-tree failed: {read.stderr.strip()}")
                    return
                add = self._git_id("add", "--", *rel_paths, extra_env=env)
                if add.returncode != 0:
                    print(f"[results] git add failed: {add.stderr.strip()}")
                    return
                tree = self._git("write-tree", extra_env=env).stdout.strip()
                if not tree:
                    print("[results] write-tree produced nothing; skipping push")
                    return
                # Nothing changed relative to the base: don't push an empty commit.
                base_tree = self._git("rev-parse", f"{base}^{{tree}}").stdout.strip()
                if tree == base_tree:
                    print("[results] results identical to the remote; nothing to push")
                    return

                commit = self._git_id("commit-tree", tree, "-p", base, "-m", msg)
                sha = commit.stdout.strip()
                if commit.returncode != 0 or not sha:
                    print(f"[results] commit-tree failed: {commit.stderr.strip()}")
                    return

                if self._push(f"{sha}:refs/heads/{branch}", branch) is True:
                    print(
                        f"[results] published {self.run_name} as {sha[:8]} "
                        "(checkout untouched)"
                    )
                    return
                if attempt < self.retries:
                    self._backoff(attempt)
        print("[results] giving up on push; results remain on disk in the run directory")

    # -- shared push mechanics ---------------------------------------------- #

    def _should_push(self) -> bool:
        if not self.push:
            print(
                "[results] not pushing (pass --push_results or set "
                "BLT_LCM_PUSH_RESULTS=1 to push automatically)"
            )
            return False
        remotes = self._git("remote").stdout.split()
        if self.remote not in remotes:
            print(f"[results] no remote named '{self.remote}'; skipping push")
            return False
        return True

    def _target_branch(self) -> Optional[str]:
        branch = self.branch or _current_branch(self.repo_root or ".")
        if not branch:
            print(
                "[results] detached HEAD and no --results_branch; "
                "cannot decide where to push"
            )
        return branch

    def _remote_url(self) -> str:
        return self._git("remote", "get-url", self.remote).stdout.strip()

    def _remote_tip(self, branch: str) -> Optional[str]:
        """Fetch the remote branch tip, so the new commit stacks on it."""
        url = self._remote_url()
        auth = _authenticated_url(url)
        target = auth or self.remote
        fetch = self._git("fetch", "--quiet", target, branch)
        if fetch.returncode != 0:
            return None
        return self._git("rev-parse", "FETCH_HEAD").stdout.strip() or None

    def _push(self, refspec: str, branch: str) -> bool:
        """One push attempt. Returns True on success; never raises."""
        url = self._remote_url()
        auth = _authenticated_url(url)
        target = auth or self.remote
        push = self._git("push", target, refspec)
        if push.returncode == 0:
            print(f"[results] pushed {self.run_name} to {self.remote}/{branch}")
            return True
        # The token must never reach a log file.
        err = (push.stderr or push.stdout).strip()
        if auth:
            err = err.replace(auth, f"{self.remote} (authenticated)")
        if push.returncode == 124:
            print(f"[results] push timed out after {self.timeout:g}s")
        elif "Authentication" in err or "could not read Username" in err:
            print(
                "[results] push failed: no usable credentials on this node. "
                "Set GITHUB_TOKEN (or GIT_TOKEN) in the job environment, or use "
                "an ssh remote with a key the compute nodes can read."
            )
        else:
            print(f"[results] push failed: {err}")
        return False

    def _backoff(self, attempt: int) -> None:
        """Randomized backoff, so an array of jobs does not retry in lockstep."""
        delay = min(2.0**attempt, 30.0) * (0.5 + random.random())
        print(f"[results] retrying push in {delay:.1f}s (attempt {attempt})")
        time.sleep(delay)
