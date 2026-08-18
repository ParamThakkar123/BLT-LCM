"""Tests for publishing the auto_setup driver's state directory.

``lcm_scripts/publish_state.py`` mirrors ``runs/auto_setup`` -- the per-step
logs, completion markers, copied artifacts and manifest -- into a directory the
repository tracks, and commits it. These tests drive it against a throwaway git
repository in ``tmp_path``; the "push" test pushes into a local bare repo
standing in for the remote, so nothing here touches the real checkout or the
network.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))

from publish_state import main, redact, tail_lines  # noqa: E402


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git("init", "-b", "main", cwd=root)
    git("config", "user.email", "test@example.com", cwd=root)
    git("config", "user.name", "Test", cwd=root)
    (root / "README.md").write_text("seed\n", encoding="utf-8")
    git("add", "-A", cwd=root)
    git("commit", "-m", "seed", cwd=root)
    return root


@pytest.fixture
def state(tmp_path):
    """A state directory shaped like the one auto_setup.sh writes."""
    root = tmp_path / "runs" / "auto_setup"
    (root / "logs").mkdir(parents=True)
    (root / "state").mkdir()
    (root / "artifacts" / "mt_f0.25_s42").mkdir(parents=True)
    (root / "remote" / "mt_f0.25_s42").mkdir(parents=True)

    (root / "logs" / "mt_f0.25_s42.log").write_text(
        "### cmd: train_lcm_blt_mt.py\nepoch 1 loss 0.5\n", encoding="utf-8"
    )
    (root / "state" / "mt_f0.25_s42.done").write_text(
        "finished 2026-08-16T12:00:00+0000 in 900s\n", encoding="utf-8"
    )
    (root / "artifacts" / "mt_f0.25_s42" / "blt_lcm_mt_0.25_s42.csv").write_text(
        "model,noise,BLEU\nblt_lcm_mt,0.0,10.0\n", encoding="utf-8"
    )
    (root / "manifest.jsonl").write_text(
        json.dumps({"id": "mt:f0.25_s42", "status": "ok", "elapsed_s": 900,
                    "batch_size": "32", "finished": "2026-08-16T12:00:00+0000"})
        + "\n",
        encoding="utf-8",
    )
    (root / "remote" / "mt_f0.25_s42" / "run.json").write_text("{}", encoding="utf-8")
    return root


def publish(state, repo, *extra):
    return main([
        "--state_dir", str(state), "--repo_root", str(repo),
        "--name", "testnode", *extra,
    ])


def mirror(repo, *parts):
    return repo.joinpath("results", "auto_setup", "testnode", *parts)


# --------------------------------------------------------------------------- #
# What goes in
# --------------------------------------------------------------------------- #


def test_the_whole_tree_is_mirrored(state, repo):
    publish(state, repo)

    assert mirror(repo, "logs", "mt_f0.25_s42.log").exists()
    assert mirror(repo, "state", "mt_f0.25_s42.done").exists()
    assert mirror(repo, "artifacts", "mt_f0.25_s42",
                  "blt_lcm_mt_0.25_s42.csv").exists()
    assert mirror(repo, "manifest.jsonl").exists()


def test_the_remote_scratch_directory_is_not_published(state, repo):
    """Copies of records that are already in git, re-fetched on every run."""
    publish(state, repo)

    assert not mirror(repo, "remote").exists()


def test_checkpoints_are_named_but_not_published(state, repo):
    (state / "artifacts" / "mt_f0.25_s42" / "best.pth").write_bytes(b"\0\1\2" * 100)
    publish(state, repo)

    assert not mirror(repo, "artifacts", "mt_f0.25_s42", "best.pth").exists()
    withheld = mirror(repo, "WITHHELD.txt").read_text(encoding="utf-8")
    assert "best.pth" in withheld and "not a publishable file type" in withheld


def test_a_figure_is_published(state, repo):
    png = state / "artifacts" / "mt_f0.25_s42" / "curve.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    publish(state, repo)

    assert mirror(repo, "artifacts", "mt_f0.25_s42", "curve.png").read_bytes() \
        == png.read_bytes()


def test_an_oversized_log_is_published_as_its_tail(state, repo):
    log = state / "logs" / "huge.log"
    log.write_text("".join(f"line {i}\n" for i in range(200_000)), encoding="utf-8")
    publish(state, repo, "--max_mb", "0.1", "--tail_lines", "50")

    text = mirror(repo, "logs", "huge.log").read_text(encoding="utf-8")
    assert "TRUNCATED by publish_state.py" in text
    assert "line 199999" in text
    assert "line 100000" not in text
    assert "huge.log" in mirror(repo, "WITHHELD.txt").read_text(encoding="utf-8")


def test_an_oversized_binary_is_withheld(state, repo):
    blob = state / "artifacts" / "mt_f0.25_s42" / "big.png"
    blob.write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(200_000))
    publish(state, repo, "--max_mb", "0.1")

    assert not mirror(repo, "artifacts", "mt_f0.25_s42", "big.png").exists()
    assert "big.png" in mirror(repo, "WITHHELD.txt").read_text(encoding="utf-8")


def test_tail_lines_reads_only_the_end(tmp_path):
    path = tmp_path / "log"
    path.write_text("".join(f"{i}\n" for i in range(5000)), encoding="utf-8")

    assert tail_lines(str(path), 3).splitlines() == ["4997", "4998", "4999"]


# --------------------------------------------------------------------------- #
# Credentials never reach the commit
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "secret",
    [
        "https://x-access-token:ghp_0123456789abcdefghijABCDEFGHIJ@github.com/o/r",
        "GITHUB_TOKEN=ghp_0123456789abcdefghijABCDEFGHIJ",
        "token: github_pat_11ABCDEFG0123456789_abcdefghij",
        "HF_TOKEN=hf_abcdefghijABCDEFGHIJ0123456789",
    ],
)
def test_tokens_are_masked(secret):
    masked, hits = redact(f"prefix {secret} suffix")

    assert hits >= 1
    assert "ghp_0123456789abcdefghijABCDEFGHIJ" not in masked
    assert "github_pat_11ABCDEFG0123456789_abcdefghij" not in masked
    assert "hf_abcdefghijABCDEFGHIJ0123456789" not in masked
    assert masked.startswith("prefix ") and masked.endswith(" suffix")


def test_a_token_in_a_log_is_masked_before_committing(state, repo, capsys):
    (state / "logs" / "push.log").write_text(
        "remote: https://x-access-token:ghp_0123456789abcdefghijABCDEFGHIJ@github.com/o/r\n",
        encoding="utf-8",
    )
    publish(state, repo)

    published = mirror(repo, "logs", "push.log").read_text(encoding="utf-8")
    assert "ghp_" not in published
    assert "***" in published
    assert "masked" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The summary, and keeping the mirror honest
# --------------------------------------------------------------------------- #


def test_the_summary_reports_every_step(state, repo):
    publish(state, repo)

    record = json.loads(mirror(repo, "state.json").read_text(encoding="utf-8"))
    assert record["machine"] == "testnode"
    assert record["status_counts"] == {"ok": 1}
    assert record["steps"]["mt:f0.25_s42"]["elapsed_s"] == 900
    assert "mt:f0.25_s42" in mirror(repo, "README.md").read_text(encoding="utf-8")


def test_only_the_last_line_of_a_step_counts(state, repo):
    """A step that failed and was then re-run reports as the re-run left it."""
    with open(state / "manifest.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "mt:f0.25_s42", "status": "failed"}) + "\n")
        fh.write(json.dumps({"id": "mt:f0.25_s42", "status": "ok"}) + "\n")
    publish(state, repo)

    record = json.loads(mirror(repo, "state.json").read_text(encoding="utf-8"))
    assert record["status_counts"] == {"ok": 1}


def test_a_marker_removed_from_the_state_directory_is_unpublished(state, repo):
    publish(state, repo)
    assert mirror(repo, "state", "mt_f0.25_s42.done").exists()

    (state / "state" / "mt_f0.25_s42.done").unlink()
    publish(state, repo)

    assert not mirror(repo, "state", "mt_f0.25_s42.done").exists()
    # And the removal is in the commit, not only on disk.
    assert "state/mt_f0.25_s42.done" in git(
        "show", "--name-status", "--format=", "HEAD", cwd=repo
    ).stdout


def test_restored_steps_are_named_in_the_summary(state, repo):
    """A cell another machine ran is finished work; the record has to say so."""
    (state / "artifacts" / "mt_f0.25_s42" / "published_from.txt").write_text(
        "run lcm_blt_mt_fraction0.25_s42\n"
        "ref origin/main\n"
        "restored 2026-08-16T13:00:00+0000\n"
        "  blt_lcm_mt_0.25_s42.csv\n"
        "  run.json\n",
        encoding="utf-8",
    )
    publish(state, repo)

    record = json.loads(mirror(repo, "state.json").read_text(encoding="utf-8"))
    entry = record["restored"]["mt_f0.25_s42"]
    assert entry["ref"] == "origin/main"
    assert entry["run"] == "lcm_blt_mt_fraction0.25_s42"
    assert entry["files"] == 2
    assert "Restored from published results" in mirror(
        repo, "README.md"
    ).read_text(encoding="utf-8")


def test_a_junk_manifest_line_does_not_stop_the_publish(state, repo):
    with open(state / "manifest.jsonl", "a", encoding="utf-8") as fh:
        fh.write("this is not json\n")
    publish(state, repo)

    assert mirror(repo, "state.json").exists()


# --------------------------------------------------------------------------- #
# Committing and pushing
# --------------------------------------------------------------------------- #


def test_the_mirror_is_committed(state, repo):
    publish(state, repo)

    committed = git("show", "--name-only", "--format=", "HEAD", cwd=repo).stdout
    assert "results/auto_setup/testnode/manifest.jsonl" in committed
    assert git("log", "-1", "--format=%s", cwd=repo).stdout.startswith(
        "state: auto_setup on testnode"
    )


def test_unrelated_working_tree_changes_are_not_committed(state, repo):
    (repo / "README.md").write_text("edited by the user\n", encoding="utf-8")
    publish(state, repo)

    committed = git("show", "--name-only", "--format=", "HEAD", cwd=repo).stdout
    assert "README.md" not in committed.replace("results/auto_setup/testnode/README.md", "")


def test_a_dry_run_commits_nothing(state, repo):
    publish(state, repo, "--dry_run")

    assert mirror(repo, "manifest.jsonl").exists()
    assert git("log", "-1", "--format=%s", cwd=repo).stdout.strip() == "seed"


def test_push_lands_on_the_remote(state, repo, tmp_path):
    remote = tmp_path / "remote.git"
    git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    git("remote", "add", "origin", str(remote), cwd=repo)
    git("push", "-u", "origin", "main", cwd=repo)

    publish(state, repo, "--push")

    listing = git("ls-tree", "-r", "--name-only", "main", cwd=remote).stdout
    assert "results/auto_setup/testnode/manifest.jsonl" in listing


def test_a_missing_state_directory_is_not_an_error(repo, tmp_path, capsys):
    assert main(["--state_dir", str(tmp_path / "nope"), "--repo_root", str(repo)]) == 0
    assert "does not exist" in capsys.readouterr().out


def test_outside_a_git_repository_it_is_a_no_op(state, tmp_path, capsys, monkeypatch):
    """No repository anywhere above the state directory: say so, change nothing."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)

    assert main(["--state_dir", str(state)]) == 0
    assert "not inside a git repository" in capsys.readouterr().out


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
