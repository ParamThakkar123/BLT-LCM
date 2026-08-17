"""Tests for the experiment driver's resume decisions.

``scripts/auto_setup.sh`` skips a step when its results already exist -- on this
disk, or pushed to the results refs by another machine. These tests drive the
real script against a throwaway checkout whose "GitHub" is a local bare repo, so
nothing here touches the working copy, the network, or a GPU.

Steps are launched through ``UV_RUN_OVERRIDE=echo``: the driver's decisions are
what is under test, not the training scripts, so a step that runs simply echoes
its command line and exits 0.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

BASH = shutil.which("bash") or ""
pytestmark = pytest.mark.skipif(not BASH, reason="the driver is a bash script")

DRIVER = os.path.join(os.path.dirname(__file__), "..", "scripts", "auto_setup.sh")

# The grid cell every test uses: fraction 0.25, seed 42, published as this run.
MT_STEP = "mt:f0.25_s42"
MT_RUN = "lcm_blt_mt_fraction0.25_s42"
MT_CSV = "blt_lcm_mt_0.25_s42.csv"


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )


def mt_record(*, epochs=3, epochs_run=3, seed=42, data_seed=42, fraction=0.25,
              finished="2026-08-16T12:00:00Z"):
    """A published record shaped like the one train_lcm_blt_mt.py writes."""
    return {
        "run_name": MT_RUN,
        "script": "train_lcm_blt_mt.py",
        "finished_utc": finished,
        "hyperparameters": {
            "epochs": epochs,
            "fraction": fraction,
            "seed": seed,
            "data_seed": data_seed,
            "noise_levels": [0.0, 0.1, 0.2],
            "entropy_model": "patching_scratch/entropy_model_marathi.pt",
        },
        "metrics": {"epochs_run": epochs_run},
        "info": {"mode": "fixed", "stop_reason": "reached the --epochs cap of %d" % epochs},
        "environment": {"git_commit": "deadbee", "git_branch": "main"},
    }


NOISE_CSV = "model,fraction,noise,seed,BLEU\n" + "".join(
    f"blt_lcm_mt,0.25,{n},42,10.0\n" for n in (0.0, 0.1, 0.2)
)


@pytest.fixture
def checkout(tmp_path):
    """A BLT-LCM-shaped checkout with a bare 'origin' it can fetch from."""
    repo = tmp_path / "repo"
    (repo / "lcm_scripts").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='blt-lcm'\n", encoding="utf-8")
    (repo / "scripts").mkdir()
    shutil.copy(DRIVER, repo / "scripts" / "auto_setup.sh")

    git("init", "-b", "main", cwd=repo)
    git("config", "user.email", "test@example.com", cwd=repo)
    git("config", "user.name", "Test", cwd=repo)
    git("add", "-A", cwd=repo)
    git("commit", "-m", "seed", cwd=repo)

    remote = tmp_path / "remote.git"
    git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    git("remote", "add", "origin", str(remote), cwd=repo)
    git("push", "-u", "origin", "main", cwd=repo)

    # Stage 2 refuses to start without the decoder and pooler on disk; the tests
    # that care about the checkpoints being absent remove them explicitly.
    (repo / "lcm_models").mkdir()
    (repo / "lcm_models" / "blt_decoder.pth").write_bytes(b"fake")
    (repo / "lcm_models" / "blt_pooler.pth").write_bytes(b"fake")
    return repo


def publish(repo, run_name, record, files=None, *, branch="main"):
    """Push a run record to the remote WITHOUT putting it in the checkout.

    A separate clone stands in for the machine that produced the results, which
    is the situation being tested: the record exists on GitHub and nowhere on
    the driver's disk.
    """
    publisher = repo.parent / f"publisher-{branch}"
    if not publisher.exists():
        git("clone", "--branch", "main", str(repo.parent / "remote.git"),
            str(publisher), cwd=repo.parent)
        git("config", "user.email", "pub@example.com", cwd=publisher)
        git("config", "user.name", "Publisher", cwd=publisher)
        if branch != "main":
            git("checkout", "-b", branch, cwd=publisher)

    out = publisher / "results" / "runs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    for name, text in (files or {}).items():
        (out / name).write_text(text, encoding="utf-8")
    git("add", "-A", cwd=publisher)
    git("commit", "-m", f"results: {run_name}", cwd=publisher)
    git("push", "origin", branch, cwd=publisher)


def drive(repo, *args, **env):
    """Run the driver over this checkout and return its combined output."""
    state = repo.parent / "state"
    (state / "state").mkdir(parents=True, exist_ok=True)
    # The dependency install is not what these tests are about.
    (state / "state" / "setup_deps.done").write_text("pretend", encoding="utf-8")

    environment = dict(os.environ)
    environment.update(
        {
            "REPO_DIR": str(repo),
            "STATE_DIR": str(state),
            "UV_RUN_OVERRIDE": "echo",
            "VRAM_MB": "8000",
            # Local bare remote, so the fetch is real but offline.
            "RESULTS_FETCH": "1",
        }
    )
    environment.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        [BASH, "scripts/auto_setup.sh", *args],
        cwd=str(repo), capture_output=True, text=True, env=environment,
    )
    return proc.stdout + proc.stderr


def ran(output, step=MT_STEP):
    """Did the driver actually launch the step's command?"""
    return "train_lcm_blt_mt.py" in output if step == MT_STEP else "blt_decoder.py" in output


# --------------------------------------------------------------------------- #
# Published results decide what is still pending
# --------------------------------------------------------------------------- #


def test_a_published_cell_is_not_run_again(checkout):
    publish(checkout, MT_RUN, mt_record(), {MT_CSV: NOISE_CSV})
    out = drive(checkout, "--only", MT_STEP)

    assert "published on origin/main" in out
    assert not ran(out)
    marker = checkout.parent / "state" / "state" / "mt_f0.25_s42.done"
    assert "published on origin/main" in marker.read_text(encoding="utf-8")


def test_a_published_cell_short_of_its_epochs_is_run(checkout):
    publish(checkout, MT_RUN, mt_record(epochs_run=1), {MT_CSV: NOISE_CSV})
    out = drive(checkout, "--only", MT_STEP)

    assert "1/3 epochs trained" in out
    assert ran(out)


def test_a_published_cell_missing_noise_levels_is_run(checkout):
    """The CSV is written after the last noise level, so a short one is partial."""
    partial = "model,fraction,noise,seed,BLEU\nblt_lcm_mt,0.25,0.0,42,10.0\n"
    publish(checkout, MT_RUN, mt_record(), {MT_CSV: partial})
    out = drive(checkout, "--only", MT_STEP)

    assert "1/3 result rows" in out
    assert ran(out)


def test_a_published_cell_for_another_configuration_is_ignored(checkout):
    publish(checkout, MT_RUN, mt_record(data_seed=7), {MT_CSV: NOISE_CSV})
    out = drive(checkout, "--only", MT_STEP)

    assert "different configuration" in out and "data_seed" in out
    assert ran(out)


def test_a_published_record_without_its_metrics_csv_is_ignored(checkout):
    publish(checkout, MT_RUN, mt_record())
    out = drive(checkout, "--only", MT_STEP)

    assert f"carries no {MT_CSV}" in out
    assert ran(out)


def test_the_newest_record_across_refs_wins(checkout):
    """A later run that did not finish must not be masked by an older one."""
    publish(checkout, MT_RUN, mt_record(finished="2026-01-01T00:00:00Z"),
            {MT_CSV: NOISE_CSV})
    publish(checkout, MT_RUN,
            mt_record(epochs_run=1, finished="2026-09-09T00:00:00Z"),
            {MT_CSV: NOISE_CSV}, branch="experiment")
    out = drive(checkout, "--only", MT_STEP,
                RESULTS_REFS="origin/main origin/experiment")

    assert "1/3 epochs trained" in out
    assert ran(out)


def test_published_results_can_be_ignored(checkout):
    publish(checkout, MT_RUN, mt_record(), {MT_CSV: NOISE_CSV})
    out = drive(checkout, "--only", MT_STEP, REMOTE_RESULTS=0)

    assert "published on origin/main" not in out
    assert ran(out)


def test_an_unreachable_remote_is_not_fatal(checkout):
    """A driver that cannot fetch still runs; it just knows less."""
    git("remote", "set-url", "origin", str(checkout.parent / "gone.git"), cwd=checkout)
    out = drive(checkout, "--only", MT_STEP)

    assert ran(out)
    assert "could not fetch" in out


# --------------------------------------------------------------------------- #
# Published record != the files this machine needs
# --------------------------------------------------------------------------- #


def decoder_record():
    return {
        "run_name": "blt_decoder",
        "script": "blt_decoder.py",
        "finished_utc": "2026-08-16T10:00:00Z",
        "hyperparameters": {
            "epochs": 10,
            "num_sentences": 50000,
            "entropy_model": "patching_scratch/entropy_model_marathi.pt",
        },
        "metrics": {"epochs_run": 10},
        "info": {"mode": "fixed"},
        "environment": {"git_commit": "deadbee"},
    }


def test_a_producer_step_still_runs_when_its_checkpoints_are_absent(checkout):
    """The decoder's outputs are never published, and Stage 2 loads them."""
    for name in ("blt_decoder.pth", "blt_pooler.pth"):
        (checkout / "lcm_models" / name).unlink()
    publish(checkout, "blt_decoder", decoder_record())
    out = drive(checkout, "--only", "decoder")

    assert "never published and are missing here" in out
    assert "blt_decoder.py" in out


def test_a_producer_step_is_skipped_when_its_checkpoints_are_here(checkout):
    publish(checkout, "blt_decoder", decoder_record())
    out = drive(checkout, "--only", "decoder")

    assert "published on origin/main" in out
    assert "blt_decoder.py" not in out


def test_trusting_producers_skips_it_anyway(checkout):
    for name in ("blt_decoder.pth", "blt_pooler.pth"):
        (checkout / "lcm_models" / name).unlink()
    publish(checkout, "blt_decoder", decoder_record())
    out = drive(checkout, "--only", "decoder", TRUST_REMOTE_PRODUCERS=1)

    assert "published on origin/main" in out
    assert "blt_decoder.py" not in out


def test_strict_mode_rejects_a_record_from_unknown_code(checkout):
    publish(checkout, MT_RUN, mt_record(), {MT_CSV: NOISE_CSV})
    out = drive(checkout, "--only", MT_STEP, REMOTE_STRICT=1)

    # The record names a commit this checkout does not contain.
    assert "not an ancestor of HEAD" in out
    assert ran(out)


def test_strict_mode_accepts_a_record_from_code_we_have(checkout):
    head = git("rev-parse", "HEAD", cwd=checkout).stdout.strip()
    record = mt_record()
    record["environment"]["git_commit"] = head
    publish(checkout, MT_RUN, record, {MT_CSV: NOISE_CSV})
    out = drive(checkout, "--only", MT_STEP, REMOTE_STRICT=1)

    assert "published on origin/main" in out
    assert not ran(out)


# --------------------------------------------------------------------------- #
# The marker still wins, and --force overrides everything
# --------------------------------------------------------------------------- #


def test_force_reruns_a_published_cell(checkout):
    publish(checkout, MT_RUN, mt_record(), {MT_CSV: NOISE_CSV})
    out = drive(checkout, "--only", MT_STEP, "--force")

    assert ran(out)


def test_a_published_cell_is_only_looked_up_once(checkout):
    """The second run skips on the driver's own marker, without any git call."""
    publish(checkout, MT_RUN, mt_record(), {MT_CSV: NOISE_CSV})
    drive(checkout, "--only", MT_STEP)
    out = drive(checkout, "--only", MT_STEP)

    assert "already done" in out
    assert not ran(out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
