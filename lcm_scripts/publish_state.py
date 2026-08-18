"""Publish the auto_setup driver's state directory to the repository.

``scripts/auto_setup.sh`` keeps everything it knows about a pipeline run under
``runs/auto_setup``: one log per step, a completion marker per step, the copied
artifacts of every step, the GPU probe, and ``manifest.jsonl``. ``runs/`` is
gitignored -- deliberately, because it is also where multi-gigabyte checkpoints
and benchmark scratch live -- so all of that has always stayed on the machine
that ran the pipeline. This module mirrors it into a *published* location the
repository does track (``results/auto_setup/<machine>/``) and commits it, so a
run's logs and markers survive the node they were produced on.

What it does, and why each rule is there:

* **One directory per machine.** The mirror lives under a name that defaults to
  the hostname, so two machines working the same grid publish side by side
  instead of overwriting each other's logs. It is *not* restored into
  ``runs/auto_setup`` by a ``git pull`` either, which matters: a ``.done``
  marker written on another machine would make this driver skip a step whose
  checkpoints do not exist here.
* **Text is truncated, not dropped.** A training log can be hundreds of MB.
  Anything over the size cap is published as its last ``--tail_lines`` lines
  with a header saying so, because the tail is the part anybody reads. Binary
  files over the cap are withheld outright -- git history is permanent.
* **Secrets are redacted.** These logs are pushed to a repository. Tokens that
  reached a log line (a git URL with credentials in it, a ``GITHUB_TOKEN=`` in
  a traceback of the environment) are masked before the file is committed.
* **Everything withheld is named.** Same contract as ``results_sync``: the
  published directory carries ``WITHHELD.txt`` listing what did not go in and
  why, so "where is the rest of the log?" has an answer.

The commit and push themselves are :class:`results_sync.GitPublisher` -- the
same non-interactive, identity-tolerant, retrying, shared-checkout-safe path
every training job uses to publish its results.

Usage (the driver calls this for you; ``--push`` is what auto_setup passes when
``BLT_LCM_PUSH_RESULTS=1``)::

    python lcm_scripts/publish_state.py --state_dir runs/auto_setup --push
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
import time
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from results_sync import (  # noqa: E402
    DEFAULT_GIT_TIMEOUT,
    GitPublisher,
    _repo_root,
    on_scheduler,
)

# Read as text, and therefore redactable and truncatable. Everything the driver
# writes itself is in here; `` `` covers the extensionless marker files.
TEXT_SUFFIXES = (
    ".log", ".txt", ".json", ".jsonl", ".csv", ".md", ".done", ".err", ".out",
    ".yaml", ".yml", ".ini", ".cfg", ".tsv", ".sh", "",
)

# Published as-is when they fit under the cap: the figures a step copied into
# its artifacts directory.
BINARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf", ".svg")

# Subdirectories of the state directory that are never published.
#   remote/  scratch copies of run records that are already in git -- the whole
#            directory is re-fetched from the results refs on every run.
SKIP_DIRS = ("remote",)

# Masked before anything is committed. Ordered most-specific first; each entry
# is (pattern, replacement).
REDACTIONS = (
    # https://user:token@host/... -- what _authenticated_url builds.
    (re.compile(r"(https://)[^/\s:@]+:[^/\s@]+(@)"), r"\1***:***\2"),
    # GitHub tokens, in every shape GitHub issues them.
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{16,})"), "***redacted-token***"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "***redacted-token***"),
    # Hugging Face and Weights & Biases.
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "***redacted-token***"),
    (re.compile(r"\b[0-9a-f]{40}\b(?=.*wandb)", re.IGNORECASE), "***redacted-token***"),
    # KEY=value / "key": "value" for anything that names itself a credential.
    (
        re.compile(
            r"(?i)\b(token|password|passwd|secret|api[-_]?key)\b(\"?\s*[:=]\s*\"?)"
            r"([^\s\"',]{6,})"
        ),
        r"\1\2***redacted***",
    ),
)


def redact(text: str) -> tuple[str, int]:
    """Mask credentials in a text blob. Returns the text and how many hits."""
    hits = 0
    for pattern, replacement in REDACTIONS:
        text, n = pattern.subn(replacement, text)
        hits += n
    return text, hits


def is_text(path: str) -> bool:
    """Text if the name says so, or if the first block decodes without NULs."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix in TEXT_SUFFIXES:
        return True
    if suffix in BINARY_SUFFIXES:
        return False
    try:
        with open(path, "rb") as fh:
            head = fh.read(8192)
    except OSError:
        return False
    if b"\0" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def tail_lines(path: str, count: int) -> str:
    """The last ``count`` lines of a file, read without loading all of it."""
    block, data, size = 65536, b"", os.path.getsize(path)
    with open(path, "rb") as fh:
        pos = size
        while pos > 0 and data.count(b"\n") <= count:
            pos = max(0, pos - block)
            fh.seek(pos)
            data = fh.read(size - pos)
    text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-count:])


class StateMirror:
    """Copies the state directory into the published tree, filtering as it goes."""

    def __init__(self, state_dir: str, dest_dir: str, *, max_bytes: float, tail: int):
        self.state_dir = state_dir
        self.dest_dir = dest_dir
        self.max_bytes = max_bytes
        self.tail = tail
        self.copied: list[str] = []
        self.truncated: list[tuple[str, int]] = []
        self.withheld: list[tuple[str, str]] = []
        self.redacted = 0

    # -- selection ---------------------------------------------------------- #

    def _sources(self):
        for root, dirs, names in os.walk(self.state_dir):
            rel_root = os.path.relpath(root, self.state_dir)
            if rel_root == ".":
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                rel_root = ""
            dirs.sort()
            for name in sorted(names):
                rel = os.path.join(rel_root, name) if rel_root else name
                yield os.path.join(root, name), rel.replace(os.sep, "/")

    # -- copying ------------------------------------------------------------ #

    def run(self) -> None:
        for src, rel in self._sources():
            try:
                size = os.path.getsize(src)
            except OSError as e:
                self.withheld.append((rel, f"unreadable ({e.strerror or e})"))
                continue
            dst = os.path.join(self.dest_dir, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if is_text(src):
                self._copy_text(src, dst, rel, size)
            elif size > self.max_bytes:
                self.withheld.append(
                    (rel, f"binary and larger than {self.max_bytes / 1e6:g} MB")
                )
            elif os.path.splitext(rel)[1].lower() in BINARY_SUFFIXES:
                shutil.copy2(src, dst)
                self.copied.append(rel)
            else:
                self.withheld.append((rel, "not a publishable file type"))

    def _copy_text(self, src: str, dst: str, rel: str, size: int) -> None:
        if size > self.max_bytes:
            body = tail_lines(src, self.tail)
            header = (
                f"### TRUNCATED by publish_state.py: {size} bytes on disk, "
                f"over the {self.max_bytes / 1e6:g} MB cap.\n"
                f"### The last {self.tail} lines follow; the full file is on the "
                f"machine that produced it, at {rel}.\n\n"
            )
            text = header + body
            self.truncated.append((rel, size))
        else:
            with open(src, "rb") as fh:
                text = fh.read().decode("utf-8", errors="replace")
        text, hits = redact(text)
        self.redacted += hits
        with open(dst, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        self.copied.append(rel)

    # -- pruning ------------------------------------------------------------ #

    def prune(self) -> list[str]:
        """Drop published files the state directory no longer has.

        The mirror has to be able to shrink: a step re-run with ``--force`` has
        its marker deleted, and a published copy that lingers would say the step
        is finished when the driver no longer thinks so.
        """
        keep = {p.replace("/", os.sep) for p in self.copied} | {
            "WITHHELD.txt", "README.md", "state.json"
        }
        removed = []
        for root, dirs, names in os.walk(self.dest_dir, topdown=False):
            dirs[:] = [d for d in dirs if d != ".git"]
            for name in names:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, self.dest_dir)
                if rel not in keep:
                    os.remove(path)
                    removed.append(rel.replace(os.sep, "/"))
            if root != self.dest_dir and not os.listdir(root):
                os.rmdir(root)
        return removed


# --------------------------------------------------------------------------- #
# the published summary
# --------------------------------------------------------------------------- #


def read_manifest(state_dir: str) -> list[dict]:
    """Every step line the driver appended, newest last. Junk lines are skipped."""
    path = os.path.join(state_dir, "manifest.jsonl")
    steps = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    steps.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return steps


def latest_by_step(steps: list[dict]) -> dict[str, dict]:
    """One entry per step id -- the last thing that happened to it."""
    latest: dict[str, dict] = {}
    for entry in steps:
        sid = str(entry.get("id", "")) or "?"
        latest[sid] = entry
    return latest


def read_restored(state_dir: str) -> dict[str, dict]:
    """What the driver pulled back out of the results refs, per step.

    ``auto_setup.sh`` drops a ``published_from.txt`` in a step's artifacts
    directory when it restores that step's published files -- the run it was
    published as, the ref it came from, and the file names. A pipeline whose
    cells ran on several machines is otherwise indistinguishable, in the
    published state, from one where those steps produced nothing.
    """
    restored: dict[str, dict] = {}
    art = os.path.join(state_dir, "artifacts")
    if not os.path.isdir(art):
        return restored
    for step in sorted(os.listdir(art)):
        note = os.path.join(art, step, "published_from.txt")
        if not os.path.isfile(note):
            continue
        entry = {"run": "", "ref": "", "restored": "", "files": 0}
        try:
            with open(note, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("  "):
                        entry["files"] = int(entry["files"]) + 1
                    else:
                        key, _, value = line.strip().partition(" ")
                        if key in entry:
                            entry[key] = value
        except OSError:
            continue
        restored[step] = entry
    return restored


def write_summary(mirror: StateMirror, name: str, state_dir: str) -> dict:
    """``state.json`` + ``README.md``: what this machine ran, and what it holds."""
    steps = read_manifest(state_dir)
    latest = latest_by_step(steps)
    counts: dict[str, int] = {}
    for entry in latest.values():
        status = str(entry.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1

    restored = read_restored(state_dir)
    record = {
        "machine": name,
        "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state_dir": state_dir,
        "scheduler": on_scheduler(),
        "steps": {sid: latest[sid] for sid in sorted(latest)},
        "status_counts": counts,
        "restored": restored,
        "files_published": len(mirror.copied),
        "truncated": [{"path": p, "bytes": n} for p, n in mirror.truncated],
        "withheld": [{"path": p, "reason": r} for p, r in mirror.withheld],
        "redactions": mirror.redacted,
    }
    with open(os.path.join(mirror.dest_dir, "state.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)

    lines = [
        f"# auto_setup state -- {name}",
        "",
        f"Published {record['published_utc']} from `{state_dir}` by "
        "`lcm_scripts/publish_state.py`.",
        "",
        "This is the BLT-LCM experiment driver's own state directory: one log "
        "and one completion marker per pipeline step, the artifacts each step "
        "was expected to produce, and the manifest of everything it ran. It is "
        "a record of what this machine did -- the results themselves are "
        "published separately, under `results/runs/`.",
        "",
        "## Steps",
        "",
        "| step | status | elapsed | batch size | finished |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sid in sorted(latest):
        e = latest[sid]
        lines.append(
            "| `{}` | {} | {}s | {} | {} |".format(
                sid,
                e.get("status", "?"),
                e.get("elapsed_s", "?"),
                e.get("batch_size") or "-",
                e.get("finished", "?"),
            )
        )
    if not latest:
        lines.append("| _(no steps recorded on this machine)_ | | | | |")

    if restored:
        lines += [
            "",
            "## Restored from published results",
            "",
            "These steps are finished, but not necessarily here: their results "
            "were read back out of the results refs and are included above as "
            "that step's artifacts.",
            "",
            "| step | published as | from | files |",
            "| --- | --- | --- | --- |",
        ]
        for step in sorted(restored):
            e = restored[step]
            lines.append(
                f"| `{step}` | `{e['run']}` | `{e['ref']}` | {e['files']} |"
            )

    if mirror.truncated:
        lines += ["", "## Truncated", "",
                  "Too large to commit whole; the tail is published instead.", ""]
        lines += [f"- `{p}` ({n} bytes on disk)" for p, n in sorted(mirror.truncated)]
    if mirror.withheld:
        lines += ["", "## Not published", ""]
        lines += [f"- `{p}` -- {r}" for p, r in sorted(mirror.withheld)]
    if mirror.redacted:
        lines += ["", f"{mirror.redacted} credential-shaped string(s) were masked "
                      "before committing."]
    lines.append("")
    with open(os.path.join(mirror.dest_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    withheld_path = os.path.join(mirror.dest_dir, "WITHHELD.txt")
    if mirror.withheld or mirror.truncated:
        with open(withheld_path, "w", encoding="utf-8") as fh:
            for p, n in sorted(mirror.truncated):
                fh.write(f"truncated  {p}  ({n} bytes on disk)\n")
            for p, r in sorted(mirror.withheld):
                fh.write(f"withheld   {p}  ({r})\n")
    elif os.path.exists(withheld_path):
        os.remove(withheld_path)
    return record


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def default_name() -> str:
    """A stable, filesystem-safe id for this machine."""
    raw = (
        os.environ.get("STATE_PUBLISH_ID")
        or os.environ.get("SLURMD_NODENAME")
        or socket.gethostname()
        or "unknown-host"
    )
    return re.sub(r"[^A-Za-z0-9._-]", "-", raw)[:64] or "unknown-host"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--state_dir", default="runs/auto_setup",
                   help="The driver's state directory (logs, markers, artifacts).")
    p.add_argument("--dest", default="results/auto_setup",
                   help="Repository directory the mirror is published into.")
    p.add_argument("--name", default=None,
                   help="Subdirectory of --dest for this machine "
                        "(default: $STATE_PUBLISH_ID, else the hostname).")
    p.add_argument("--repo_root", default=None,
                   help="Repository root (default: discovered from --state_dir).")
    p.add_argument("--push", action="store_true",
                   default=os.environ.get("BLT_LCM_PUSH_RESULTS", "") == "1",
                   help="Push the commit. Also enabled by BLT_LCM_PUSH_RESULTS=1.")
    p.add_argument("--remote", default=os.environ.get("BLT_LCM_RESULTS_REMOTE", "origin"))
    p.add_argument("--branch", default=os.environ.get("BLT_LCM_RESULTS_BRANCH") or None,
                   help="Branch to push to (default: the current branch).")
    p.add_argument("--max_mb", type=float, default=5.0,
                   help="Text files larger than this are published as their tail; "
                        "binaries larger than this are not published at all.")
    p.add_argument("--tail_lines", type=int, default=2000,
                   help="Lines kept from the end of an oversized text file.")
    p.add_argument("--commit_mode", choices=["auto", "isolated", "worktree"],
                   default=os.environ.get("BLT_LCM_RESULTS_COMMIT_MODE", "auto"),
                   help="'isolated' commits through a temporary index without "
                        "touching a shared checkout's HEAD; 'auto' picks it "
                        "under a scheduler.")
    p.add_argument("--retries", type=int,
                   default=int(os.environ.get("BLT_LCM_RESULTS_RETRIES", "5")))
    p.add_argument("--timeout", type=float,
                   default=float(os.environ.get("BLT_LCM_RESULTS_TIMEOUT",
                                                DEFAULT_GIT_TIMEOUT)))
    p.add_argument("--dry_run", action="store_true",
                   help="Mirror and report, but neither commit nor push.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    state_dir = os.path.abspath(args.state_dir)
    if not os.path.isdir(state_dir):
        print(f"[state] nothing to publish: {state_dir} does not exist")
        return 0

    root = args.repo_root or _repo_root(state_dir) or _repo_root(os.getcwd())
    if not root:
        print("[state] not inside a git repository; the state directory stays local")
        return 0
    root = os.path.abspath(root)

    name = args.name or default_name()
    dest_rel = f"{args.dest.rstrip('/')}/{name}"
    dest_dir = os.path.join(root, dest_rel.replace("/", os.sep))
    os.makedirs(dest_dir, exist_ok=True)

    mirror = StateMirror(
        state_dir, dest_dir, max_bytes=args.max_mb * 1024 * 1024, tail=args.tail_lines
    )
    mirror.run()
    record = write_summary(mirror, name, state_dir)
    removed = mirror.prune()

    print(
        f"[state] mirrored {len(mirror.copied)} file(s) from {state_dir} "
        f"into {dest_rel}"
    )
    if mirror.truncated:
        print(f"[state] {len(mirror.truncated)} oversized log(s) published as tails")
    if mirror.withheld:
        print(f"[state] {len(mirror.withheld)} file(s) withheld -- see "
              f"{dest_rel}/WITHHELD.txt")
    if mirror.redacted:
        print(f"[state] masked {mirror.redacted} credential-shaped string(s)")
    if removed:
        print(f"[state] dropped {len(removed)} file(s) the state directory no "
              "longer has")
    if args.dry_run:
        print("[state] --dry_run: not committing")
        return 0

    counts = record["status_counts"]
    headline = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "no steps yet"
    publisher = GitPublisher(
        repo_root=root,
        label=f"auto_setup state ({name})",
        remote=args.remote,
        branch=args.branch,
        push=args.push,
        isolated=args.commit_mode == "isolated"
        or (args.commit_mode == "auto" and on_scheduler() is not None),
        retries=args.retries,
        timeout=args.timeout,
        prefix="[state]",
    )
    message = (
        f"state: auto_setup on {name} ({headline})\n\n"
        f"driver state directory {args.state_dir} published to {dest_rel}\n"
        f"files: {len(mirror.copied)}  truncated: {len(mirror.truncated)}  "
        f"withheld: {len(mirror.withheld)}"
    )
    # One directory pathspec, not a file list: `git add <dir>` also stages the
    # deletions prune() just made, which a list of surviving files would not.
    publisher.commit_and_push([dest_rel], message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
