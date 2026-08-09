"""Tests for the shared checkpoint / resume helpers.

The property that matters is exactness: a run that is killed and resumed must
end up in the same state as one that was never interrupted. Anything weaker
(resume-at-epoch-boundary, or a sampler that reshuffles differently after a
restart) silently changes the experiment, which is worse than crashing.
"""

import json
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))

from checkpoint_utils import (  # noqa: E402
    ResumableJsonl,
    ResumableLoader,
    StageTracker,
    TrainingCheckpointer,
    cached_torch,
    capture_rng_state,
    config_fingerprint,
    load_model_state,
    restore_rng_state,
    seed_everything,
)


def _dataset(n=64, d=8):
    g = torch.Generator().manual_seed(0)
    return torch.utils.data.TensorDataset(
        torch.randn(n, d, generator=g), torch.randn(n, 1, generator=g)
    )


def _train(out_dir, epochs=3, stop_after=None, resume="auto", fingerprint="fp1"):
    """Train a toy model, optionally aborting after `stop_after` steps."""
    seed_everything(1234)
    loader = ResumableLoader(_dataset(), batch_size=8, seed=7, shuffle=True)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)

    ck = TrainingCheckpointer(
        out_dir,
        "toy",
        fingerprint=fingerprint,
        save_interval_steps=1,
        max_keep=2,
        verbose=False,
    )
    rp = ck.restore(ck.load(resume), model, opt, sched)
    step = rp.global_step
    for epoch in range(rp.start_epoch, epochs):
        for batch_idx, (x, y) in loader.epoch(epoch, skip=rp.batches_to_skip(epoch)):
            opt.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
            opt.step()
            sched.step()
            step += 1
            ck.maybe_save(
                model, opt, sched, epoch=epoch, batch_in_epoch=batch_idx,
                global_step=step,
            )
            if stop_after is not None and step >= stop_after:
                return model, step, "killed"
        ck.save_epoch(model, opt, sched, epoch=epoch, global_step=step)
    return model, step, "done"


def test_resumed_run_matches_uninterrupted_run(tmp_path):
    """A killed-and-resumed run must land on the same weights and step count."""
    reference, ref_steps, status = _train(str(tmp_path / "ref"))
    assert status == "done"

    res_dir = str(tmp_path / "res")
    _, killed_at, status = _train(res_dir, stop_after=13)
    assert status == "killed" and killed_at == 13

    resumed, res_steps, status = _train(res_dir)
    assert status == "done"
    assert res_steps == ref_steps

    for (name, a), b in zip(
        reference.state_dict().items(), resumed.state_dict().values()
    ):
        assert torch.equal(a, b), f"parameter {name} diverged after resume"


def test_resume_never_starts_from_scratch(tmp_path):
    out = str(tmp_path / "run")
    _train(out, stop_after=13)
    ck = TrainingCheckpointer(out, "toy", verbose=False)
    assert ck.load("never") is None


def test_fingerprint_mismatch_is_refused(tmp_path):
    """Resuming under changed hyperparameters must fail loudly, not silently."""
    out = str(tmp_path / "run")
    _train(out, stop_after=13)
    with pytest.raises(RuntimeError, match="different configuration"):
        _train(out, fingerprint="something-else")


def test_missing_explicit_checkpoint_raises(tmp_path):
    ck = TrainingCheckpointer(str(tmp_path), "toy", verbose=False)
    with pytest.raises(FileNotFoundError):
        ck.load(str(tmp_path / "nope.pth"))


def test_epoch_checkpoints_are_pruned_numerically(tmp_path):
    """Pruning must keep epoch 10 over epoch 9 (not sort lexicographically)."""
    out = str(tmp_path / "run")
    model = nn.Linear(2, 2)
    ck = TrainingCheckpointer(out, "toy", max_keep=2, verbose=False)
    for epoch in range(11):
        ck.save_epoch(model, epoch=epoch)
    kept = sorted(p for p in os.listdir(out) if p.startswith("toy_epoch"))
    assert kept == ["toy_epoch10.pth", "toy_epoch11.pth"]


def test_config_fingerprint_ignores_io_and_logging_flags():
    base = {"lr": 1e-4, "epochs": 3, "out_dir": "a", "wandb": False, "device": "cpu"}
    moved = {**base, "out_dir": "b", "wandb": True, "device": "cuda"}
    assert config_fingerprint(base) == config_fingerprint(moved)
    assert config_fingerprint(base) != config_fingerprint({**base, "lr": 1e-3})


def test_load_model_state_accepts_payload_and_bare_state_dict(tmp_path):
    model = nn.Linear(3, 3)
    bare = tmp_path / "bare.pth"
    torch.save(model.state_dict(), bare)
    assert "weight" in load_model_state(str(bare))

    ck = TrainingCheckpointer(str(tmp_path), "toy", verbose=False)
    ck.save(model, epoch=0)
    assert "weight" in load_model_state(ck.last_path)


def test_rng_state_round_trips():
    seed_everything(7)
    snapshot = capture_rng_state()
    first = torch.randn(4)
    restore_rng_state(snapshot)
    assert torch.equal(first, torch.randn(4))


class TestResumableJsonl:
    def test_skips_finished_records(self, tmp_path):
        path = str(tmp_path / "scan.jsonl")
        with ResumableJsonl(path, fingerprint="fp", verbose=False, key="id") as w:
            for i in range(5):
                w.append({"id": i, "v": i * i})

        w2 = ResumableJsonl(path, fingerprint="fp", verbose=False, key="id")
        assert w2.done == set(range(5))
        todo = [i for i in range(8) if not w2.is_done(i)]
        assert todo == [5, 6, 7]
        for i in todo:
            w2.append({"id": i, "v": i * i})
        w2.close()

        final = ResumableJsonl(path, fingerprint="fp", verbose=False, key="id")
        assert [r["id"] for r in final.all_records()] == list(range(8))
        final.close()

    def test_torn_trailing_line_is_dropped(self, tmp_path):
        """A process killed mid-write leaves a partial line; it must not poison
        the file or be counted as done."""
        path = str(tmp_path / "scan.jsonl")
        with ResumableJsonl(path, fingerprint="fp", verbose=False, key="id") as w:
            for i in range(3):
                w.append({"id": i})
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"id": 3, "v"')

        w2 = ResumableJsonl(path, fingerprint="fp", verbose=False, key="id")
        assert w2.done == {0, 1, 2}
        w2.append({"id": 3})
        w2.close()
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert [r["id"] for r in rows] == [0, 1, 2, 3]

    def test_changed_config_discards_partial_output(self, tmp_path):
        path = str(tmp_path / "scan.jsonl")
        with ResumableJsonl(path, fingerprint="fp1", verbose=False, key="id") as w:
            w.append({"id": 0})
        fresh = ResumableJsonl(path, fingerprint="fp2", verbose=False, key="id")
        assert fresh.done == set()
        fresh.close()


class TestStageTracker:
    def test_completed_stage_is_not_recomputed(self, tmp_path):
        path = str(tmp_path / "stages.json")
        calls = []

        def work():
            calls.append(1)
            return {"BLEU": 1.0}

        StageTracker(path, fingerprint="fp", verbose=False).run("noise=0.0", work)
        again = StageTracker(path, fingerprint="fp", verbose=False)
        assert again.run("noise=0.0", work) == {"BLEU": 1.0}
        assert len(calls) == 1

    def test_changed_config_discards_stages(self, tmp_path):
        path = str(tmp_path / "stages.json")
        StageTracker(path, fingerprint="fp1", verbose=False).set("a", 1)
        assert not StageTracker(path, fingerprint="fp2", verbose=False).done("a")

    def test_corrupt_sidecar_does_not_break_the_run(self, tmp_path):
        path = str(tmp_path / "stages.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        assert StageTracker(path, fingerprint="fp", verbose=False).stages == {}


class TestCachedTorch:
    def test_reuses_then_invalidates_on_config_change(self, tmp_path):
        path = str(tmp_path / "cache.pt")
        builds = []

        def build():
            builds.append(1)
            return torch.ones(3)

        first = cached_torch(path, build, fingerprint="fp1", verbose=False)
        second = cached_torch(path, build, fingerprint="fp1", verbose=False)
        assert len(builds) == 1 and torch.equal(first, second)

        cached_torch(path, build, fingerprint="fp2", verbose=False)
        assert len(builds) == 2

    def test_failed_validation_recomputes(self, tmp_path):
        path = str(tmp_path / "cache.pt")
        builds = []

        def build():
            builds.append(1)
            return []

        cached_torch(path, build, fingerprint="fp", validate=bool, verbose=False)
        cached_torch(path, build, fingerprint="fp", validate=bool, verbose=False)
        assert len(builds) == 2, "an empty cache should not be reused"

    def test_no_path_disables_caching(self, tmp_path):
        builds = []
        for _ in range(2):
            cached_torch(None, lambda: builds.append(1), verbose=False)
        assert len(builds) == 2
