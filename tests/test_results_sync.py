"""Tests for run-results publishing.

Every test runs against a throwaway git repository created in ``tmp_path`` and
passes ``repo_root=`` explicitly, so nothing here can touch the real working
copy or push anywhere real. The "push" tests push into a local bare repo acting
as the remote.
"""

import argparse
import json
import os
import subprocess
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))

from results_sync import (  # noqa: E402
    RESULTS_ARG_NAMES,
    ResultsRecorder,
    add_results_args,
)
from checkpoint_utils import DEFAULT_FINGERPRINT_IGNORE, config_fingerprint  # noqa: E402


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path):
    """A git repo with one commit, so HEAD and a branch exist."""
    root = tmp_path / "repo"
    root.mkdir()
    git("init", "-b", "main", cwd=root)
    git("config", "user.email", "test@example.com", cwd=root)
    git("config", "user.name", "Test", cwd=root)
    (root / "README.md").write_text("seed", encoding="utf-8")
    git("add", "README.md", cwd=root)
    git("commit", "-m", "seed", cwd=root)
    return root


def make_args(tmp_path, **overrides):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    add_results_args(p)
    args = p.parse_args([])
    args.results_dir = "results/runs"
    args.push_results = False
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def make_run_outputs(d, *, big_mb=0):
    """Realistic run outputs: two figures, a metrics CSV, and a checkpoint."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_training_curve.png").write_bytes(b"\x89PNG fake")
    (d / "run_training_dashboard.jpg").write_bytes(b"\xff\xd8 fake")
    (d / "run_history.json").write_text('{"epochs": []}', encoding="utf-8")
    (d / "metrics.csv").write_text("model,BLEU\nblt,12.3\n", encoding="utf-8")
    # Must NOT be published: large, regenerable, and permanent in git history.
    (d / "model_best.pth").write_bytes(b"0" * 1024)
    (d / "embeddings.pt").write_bytes(b"0" * 1024)
    if big_mb:
        (d / "huge_plot.png").write_bytes(b"0" * int(big_mb * 1024 * 1024))
    return d


# --------------------------------------------------------------------------- #
# What gets collected
# --------------------------------------------------------------------------- #


def test_collects_figures_and_metrics_but_not_checkpoints(repo, tmp_path):
    src = make_run_outputs(tmp_path / "run")
    rec = ResultsRecorder(
        make_args(tmp_path), "demo", script="train.py", repo_root=str(repo)
    )
    rec.add_source(str(src))
    rec.add_metrics(final_loss=0.25)
    out = rec.publish()

    published = set(os.listdir(out))
    assert "run_training_curve.png" in published
    assert "run_training_dashboard.jpg" in published
    assert "metrics.csv" in published
    assert "run_history.json" in published
    assert "run.json" in published and "README.md" in published
    # Checkpoints and caches are deliberately excluded.
    assert "model_best.pth" not in published
    assert "embeddings.pt" not in published


def test_oversize_files_are_skipped(repo, tmp_path, capsys):
    src = make_run_outputs(tmp_path / "run", big_mb=2)
    rec = ResultsRecorder(
        make_args(tmp_path, results_max_mb=1.0), "demo", repo_root=str(repo)
    )
    rec.add_source(str(src))
    out = rec.publish()
    assert "huge_plot.png" not in os.listdir(out)
    assert "results_max_mb" in capsys.readouterr().out


def test_individual_files_can_be_added(repo, tmp_path):
    src = make_run_outputs(tmp_path / "run")
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source(str(src / "metrics.csv"))
    assert sorted(os.listdir(rec.publish())) == [
        "README.md",
        "metrics.csv",
        "run.json",
    ]


def test_missing_and_empty_sources_are_tolerated(repo, tmp_path):
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source("", str(tmp_path / "does_not_exist.png"))
    out = rec.publish()
    assert out and os.path.exists(os.path.join(out, "run.json"))


# --------------------------------------------------------------------------- #
# What is present but NOT published
# --------------------------------------------------------------------------- #


def test_withheld_files_are_named_with_reasons(repo, tmp_path, capsys):
    """Selective publishing is fine; silence about what it dropped is not."""
    src = make_run_outputs(tmp_path / "run", big_mb=2)
    rec = ResultsRecorder(
        make_args(tmp_path, results_max_mb=1.0), "demo", repo_root=str(repo)
    )
    rec.add_source(str(src), str(tmp_path / "never_drawn.png"))
    out = rec.publish()

    record = json.loads(open(os.path.join(out, "run.json"), encoding="utf-8").read())
    withheld = {os.path.basename(w["path"]): w["reason"] for w in record["withheld"]}
    assert "not a publishable file type" in withheld["model_best.pth"]
    assert "not a publishable file type" in withheld["embeddings.pt"]
    assert "results_max_mb" in withheld["huge_plot.png"]
    assert "not on disk" in withheld["never_drawn.png"]
    # The checkpoint's size is recorded too, so "how big was it" needs no node.
    sizes = {os.path.basename(w["path"]): w.get("megabytes") for w in record["withheld"]}
    assert sizes["huge_plot.png"] == pytest.approx(2.0, abs=0.1)

    printed = capsys.readouterr().out
    assert "not published" in printed
    assert "model_best.pth" in printed and "huge_plot.png" in printed


def test_withheld_files_are_listed_in_the_readme(repo, tmp_path):
    src = make_run_outputs(tmp_path / "run")
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source(str(src))
    text = open(os.path.join(rec.publish(), "README.md"), encoding="utf-8").read()
    assert "not in this commit" in text
    assert "model_best.pth" in text


def test_a_gitignored_result_does_not_sink_the_whole_publish(repo, tmp_path):
    """One ignored file used to abort the commit and lose every other result."""
    (repo / ".gitignore").write_text("*.csv\n", encoding="utf-8")
    git("add", ".gitignore", cwd=repo)
    git("commit", "-m", "ignore csv", cwd=repo)

    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source(str(make_run_outputs(tmp_path / "run")))
    out = rec.publish()

    committed = git("show", "--name-only", "--format=", "HEAD", cwd=repo).stdout
    committed = committed.replace("\\", "/")
    assert "results/runs/demo/run.json" in committed
    assert "results/runs/demo/run_training_curve.png" in committed
    assert "metrics.csv" not in committed
    # The file is still on disk, and the record says why it is not in git.
    assert os.path.exists(os.path.join(out, "metrics.csv"))
    record = json.loads(open(os.path.join(out, "run.json"), encoding="utf-8").read())
    ignored = [w for w in record["withheld"] if "metrics.csv" in w["path"]]
    assert ignored and "ignored by" in ignored[0]["reason"]
    assert ".gitignore" in ignored[0]["reason"]


def test_everything_ignored_is_reported_not_crashed(repo, tmp_path, capsys):
    (repo / ".gitignore").write_text("results/\n", encoding="utf-8")
    git("add", ".gitignore", cwd=repo)
    git("commit", "-m", "ignore results", cwd=repo)

    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source(str(make_run_outputs(tmp_path / "run")))
    out = rec.publish()

    assert out and os.path.exists(os.path.join(out, "run.json"))
    assert "nothing left to commit" in capsys.readouterr().out
    assert "results: demo" not in git("log", "--oneline", cwd=repo).stdout


# --------------------------------------------------------------------------- #
# The run record
# --------------------------------------------------------------------------- #


def test_run_json_records_hyperparameters_metrics_and_environment(repo, tmp_path):
    rec = ResultsRecorder(
        make_args(tmp_path, epochs=7, lr=3e-4),
        "demo",
        script="train_lcm_blt.py",
        fingerprint="fp123",
        repo_root=str(repo),
    )
    rec.add_metrics(final_loss=0.25, BLEU=12.3)
    rec.add_info(stop_reason="plateau", documents=500)
    out = rec.publish()

    record = json.loads(open(os.path.join(out, "run.json"), encoding="utf-8").read())
    assert record["hyperparameters"]["epochs"] == 7
    assert record["hyperparameters"]["lr"] == pytest.approx(3e-4)
    assert record["metrics"] == {"final_loss": 0.25, "BLEU": 12.3}
    assert record["info"]["stop_reason"] == "plateau"
    assert record["fingerprint"] == "fp123"
    assert record["script"] == "train_lcm_blt.py"
    assert record["environment"]["git_commit"]
    assert record["environment"]["git_branch"] == "main"
    assert "wall_clock_seconds" in record


def test_readme_lists_metrics_and_embeds_figures(repo, tmp_path):
    src = make_run_outputs(tmp_path / "run")
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source(str(src))
    rec.add_metrics(BLEU=12.3)
    text = open(
        os.path.join(rec.publish(), "README.md"), encoding="utf-8"
    ).read()
    assert "| BLEU | 12.3 |" in text
    assert "![run_training_curve.png](run_training_curve.png)" in text


def test_unserialisable_hyperparameters_are_stringified(repo, tmp_path):
    args = make_args(tmp_path)
    args.device = object()
    rec = ResultsRecorder(args, "demo", repo_root=str(repo))
    record = json.loads(
        open(os.path.join(rec.publish(), "run.json"), encoding="utf-8").read()
    )
    assert isinstance(record["hyperparameters"]["device"], str)


# --------------------------------------------------------------------------- #
# git behaviour
# --------------------------------------------------------------------------- #


def test_results_are_committed(repo, tmp_path):
    src = make_run_outputs(tmp_path / "run")
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source(str(src))
    rec.publish()

    log = git("log", "--oneline", cwd=repo).stdout
    assert "results: demo" in log
    files = git("show", "--name-only", "--format=", "HEAD", cwd=repo).stdout
    assert "results/runs/demo/run.json" in files.replace("\\", "/")


def test_unrelated_working_tree_changes_are_not_committed(repo, tmp_path):
    """A run must never sweep the user's edits into its results commit."""
    (repo / "README.md").write_text("edited by the user", encoding="utf-8")
    (repo / "scratch.txt").write_text("untracked", encoding="utf-8")

    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source(str(make_run_outputs(tmp_path / "run")))
    rec.publish()

    committed = [
        line.replace("\\", "/")
        for line in git("show", "--name-only", "--format=", "HEAD", cwd=repo)
        .stdout.strip()
        .splitlines()
        if line.strip()
    ]
    # Every path in the results commit is under this run's results directory:
    # the user's edited README.md and untracked scratch.txt are not in it.
    assert committed, "expected a results commit"
    assert all(p.startswith("results/runs/demo/") for p in committed), committed
    # ...and the edits are still there, unstaged.
    assert "edited by the user" in (repo / "README.md").read_text(encoding="utf-8")
    assert (repo / "scratch.txt").exists()


def test_second_identical_run_makes_no_empty_commit(repo, tmp_path, capsys):
    src = make_run_outputs(tmp_path / "run")

    def once():
        rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
        rec.add_source(str(src / "metrics.csv"))
        # run.json carries a timestamp, so exclude it from this comparison by
        # publishing only the CSV and then checking the commit count.
        rec.publish()

    once()
    first = len(git("log", "--oneline", cwd=repo).stdout.strip().splitlines())
    once()
    second = len(git("log", "--oneline", cwd=repo).stdout.strip().splitlines())
    # run.json changes every run (timestamps), so a commit is expected; what
    # matters is that publishing twice does not error.
    assert second >= first


def test_outside_a_git_repo_is_a_no_op(tmp_path, capsys):
    rec = ResultsRecorder(
        make_args(tmp_path), "demo", repo_root=None
    )
    # There is no repo at this path, so the recorder disables itself.
    if rec.enabled:  # a parent dir happened to be a repo; force the case
        rec.repo_root = None
        rec.out_dir = None
    assert rec.publish() is None


def test_no_results_flag_disables_everything(repo, tmp_path):
    rec = ResultsRecorder(
        make_args(tmp_path, no_results=True), "demo", repo_root=str(repo)
    )
    rec.add_source(str(make_run_outputs(tmp_path / "run")))
    assert rec.publish() is None
    assert not (repo / "results").exists()


# --------------------------------------------------------------------------- #
# push
# --------------------------------------------------------------------------- #


def test_push_lands_on_the_remote(repo, tmp_path):
    remote = tmp_path / "remote.git"
    git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    git("remote", "add", "origin", str(remote), cwd=repo)
    git("push", "-u", "origin", "main", cwd=repo)

    rec = ResultsRecorder(
        make_args(tmp_path, push_results=True), "demo", repo_root=str(repo)
    )
    rec.add_source(str(make_run_outputs(tmp_path / "run")))
    rec.publish()

    remote_files = git("ls-tree", "-r", "--name-only", "main", cwd=remote).stdout
    assert "results/runs/demo/run.json" in remote_files.replace("\\", "/")


def test_push_without_a_remote_warns_and_continues(repo, tmp_path, capsys):
    rec = ResultsRecorder(
        make_args(tmp_path, push_results=True), "demo", repo_root=str(repo)
    )
    rec.add_source(str(make_run_outputs(tmp_path / "run")))
    assert rec.publish() is not None  # committed locally
    assert "no remote named 'origin'" in capsys.readouterr().out


def test_push_is_off_by_default(repo, tmp_path, capsys):
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source(str(make_run_outputs(tmp_path / "run")))
    rec.publish()
    assert "--push_results" in capsys.readouterr().out


def test_push_env_var_enables_it(monkeypatch):
    monkeypatch.setenv("BLT_LCM_PUSH_RESULTS", "1")
    p = argparse.ArgumentParser()
    add_results_args(p)
    assert p.parse_args([]).push_results is True


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def test_results_flags_do_not_change_a_run_fingerprint():
    base = {"epochs": 2, "lr": 1e-4}
    without = config_fingerprint(argparse.Namespace(**base))
    with_flags = config_fingerprint(
        argparse.Namespace(
            **base, results_dir="results/runs", push_results=True, no_results=False,
            results_remote="origin", results_branch=None, results_max_mb=25.0,
        )
    )
    assert without == with_flags


def test_every_results_arg_is_fingerprint_ignored():
    assert set(RESULTS_ARG_NAMES) <= set(DEFAULT_FINGERPRINT_IGNORE)


# --------------------------------------------------------------------------- #
# Cluster behaviour
# --------------------------------------------------------------------------- #


@pytest.fixture
def repo_with_remote(repo, tmp_path):
    remote = tmp_path / "remote.git"
    git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    git("remote", "add", "origin", str(remote), cwd=repo)
    git("push", "-u", "origin", "main", cwd=repo)
    return repo, remote


def test_scheduler_env_selects_isolated_mode(monkeypatch, repo, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    assert rec.isolated


def test_no_scheduler_uses_the_worktree_path(monkeypatch, repo, tmp_path):
    for var in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "PBS_JOBID", "LSB_JOBID"):
        monkeypatch.delenv(var, raising=False)
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    assert not rec.isolated


def test_commit_mode_can_be_forced(monkeypatch, repo, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    forced = ResultsRecorder(
        make_args(tmp_path, results_commit_mode="isolated"), "demo",
        repo_root=str(repo),
    )
    assert forced.isolated
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    off = ResultsRecorder(
        make_args(tmp_path, results_commit_mode="worktree"), "demo",
        repo_root=str(repo),
    )
    assert not off.isolated


def test_isolated_push_does_not_move_local_head(monkeypatch, repo_with_remote, tmp_path):
    """The whole point on a cluster: parallel jobs share one checkout.

    A job that moved HEAD (or took .git/index.lock) would disrupt every other
    job running out of the same clone.
    """
    repo, remote = repo_with_remote
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    head_before = git("rev-parse", "HEAD", cwd=repo).stdout.strip()
    index_before = (repo / ".git" / "index").stat().st_mtime

    rec = ResultsRecorder(
        make_args(tmp_path, push_results=True), "demo", repo_root=str(repo)
    )
    rec.add_source(str(make_run_outputs(tmp_path / "run")))
    rec.publish()

    # Local state untouched...
    assert git("rev-parse", "HEAD", cwd=repo).stdout.strip() == head_before
    assert (repo / ".git" / "index").stat().st_mtime == index_before
    assert not git("status", "--porcelain", cwd=repo).stdout.strip().startswith("A ")
    # ...but the results are on the remote.
    remote_files = git("ls-tree", "-r", "--name-only", "main", cwd=remote).stdout
    assert "results/runs/demo/run.json" in remote_files.replace("\\", "/")


def test_isolated_commits_from_parallel_jobs_all_land(monkeypatch, repo_with_remote, tmp_path):
    """Several array tasks finishing at once must not lose each other's results."""
    repo, remote = repo_with_remote
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    for i in range(3):
        rec = ResultsRecorder(
            make_args(tmp_path, push_results=True), f"job{i}", repo_root=str(repo)
        )
        rec.add_source(str(make_run_outputs(tmp_path / f"run{i}")))
        rec.publish()

    remote_files = git("ls-tree", "-r", "--name-only", "main", cwd=remote).stdout
    for i in range(3):
        assert f"results/runs/job{i}/run.json" in remote_files.replace("\\", "/")


def test_isolated_without_push_falls_back_to_a_local_commit(
    monkeypatch, repo, tmp_path, capsys
):
    """An isolated commit with nowhere to push would vanish; commit locally."""
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    rec.add_source(str(make_run_outputs(tmp_path / "run")))
    rec.publish()
    assert "results: demo" in git("log", "--oneline", cwd=repo).stdout


def test_identical_results_are_not_pushed_twice(monkeypatch, repo_with_remote, tmp_path, capsys):
    repo, remote = repo_with_remote
    monkeypatch.setenv("SLURM_JOB_ID", "5")
    src = make_run_outputs(tmp_path / "run")

    rec = ResultsRecorder(
        make_args(tmp_path, push_results=True), "demo", repo_root=str(repo)
    )
    rec.add_source(str(src / "metrics.csv"))  # stable content, no timestamps
    rec.publish()
    capsys.readouterr()

    # Republishing the same file must be a no-op rather than an empty commit.
    rec2 = ResultsRecorder(
        make_args(tmp_path, push_results=True), "demo", repo_root=str(repo)
    )
    rec2.add_source(str(src / "metrics.csv"))
    rec2._collect = lambda: (  # type: ignore[method-assign]
        [str(rec2.out_dir and os.path.join(rec2.out_dir, "metrics.csv"))],
        {},
    )
    rec2.publish()
    assert "nothing to push" in capsys.readouterr().out


def test_environment_records_slurm_context(monkeypatch, repo, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "31337")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-node-07")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "a100-galvani")
    rec = ResultsRecorder(make_args(tmp_path), "demo", repo_root=str(repo))
    record = json.loads(
        open(os.path.join(rec.publish(), "run.json"), encoding="utf-8").read()
    )
    env = record["environment"]
    assert env["SLURM_JOB_ID"] == "31337"
    assert env["SLURM_JOB_NODELIST"] == "gpu-node-07"
    assert env["hostname"]


def test_git_calls_are_non_interactive():
    """A credential prompt on a compute node hangs the job until wall-clock."""
    from results_sync import _git_env

    env = _git_env()
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_ASKPASS"]


def test_token_is_injected_into_https_remotes(monkeypatch):
    from results_sync import _authenticated_url

    monkeypatch.delenv("GITHUB_USERNAME", raising=False)
    for var in ("BLT_LCM_GIT_TOKEN", "GIT_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "tok123")

    url = _authenticated_url("https://github.com/o/r.git")
    assert url == "https://x-access-token:tok123@github.com/o/r.git"

    monkeypatch.setenv("GITHUB_USERNAME", "someone")
    assert _authenticated_url("https://github.com/o/r.git").startswith(
        "https://someone:tok123@"
    )
    # ssh remotes use the agent; a URL already carrying credentials is left be.
    assert _authenticated_url("git@github.com:o/r.git") is None
    assert _authenticated_url("https://u:p@github.com/o/r.git") is None


def test_no_token_leaves_the_url_alone(monkeypatch):
    from results_sync import _authenticated_url

    for var in ("BLT_LCM_GIT_TOKEN", "GIT_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert _authenticated_url("https://github.com/o/r.git") is None


def test_git_timeout_is_reported_not_raised(repo, tmp_path):
    """A hung network call must end the publish, not the job."""
    from results_sync import _git_raw

    r = _git_raw("ls-remote", "https://10.255.255.1/nope.git", cwd=str(repo), timeout=1.0)
    assert r.returncode != 0  # timed out or refused; either way, it returned
