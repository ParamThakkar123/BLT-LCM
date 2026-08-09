"""Fidelity tests: the implementations must match the BLT and LCM papers.

Each test cites the specific claim it pins down, so a future refactor that
silently drifts away from the papers fails here rather than in a results table.

  BLT — Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than
        Tokens" (2024).
  LCM — Barrault et al., "Large Concept Models: Language Modeling in a Sentence
        Representation Space" (2024).
"""

import gc
import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "patching_scratch"))

from run_blt_patching import (  # noqa: E402
    OFFSET,
    ByteEntropyModel,
    compute_entropies_for_tokens,
    entropy_patch_sentence,
    sliding_window_causal_mask,
    split_on_newlines,
    text_to_byte_tokens,
)
from blt_local_encoder import (  # noqa: E402
    HASH_BASE_PRIME,
    BLTLatentTransformer,
    BLTSentenceEncoder,
    HashNGramEmbedder,
    _rolling_poly_hash_tensor,
    rolling_poly_hash,
)
from base_lcm import BaseLCM, RobustScaler  # noqa: E402
from diffusion_lcm import (  # noqa: E402
    GaussianDiffusion,
    OneTowerDiffusionLCM,
    TwoTowerDiffusionLCM,
    make_noise_schedule,
)
from quant_lcm import QuantLCM, ResidualVectorQuantizer  # noqa: E402

MOD = 2**61 - 1


@pytest.fixture(autouse=True)
def _release_models():
    """Drop model allocations between tests.

    Several tests here build small transformers, and tests/test_finetune.py
    builds 1024x2048 ones. Without reclaiming between tests the combined suite
    can exhaust the Windows paging file and fail with allocator errors that
    look like logic failures.
    """
    yield
    gc.collect()


@pytest.fixture(scope="module")
def entropy_model():
    torch.manual_seed(0)
    return ByteEntropyModel(dim=64, n_heads=4, n_layers=2, max_seqlen=512).eval()


@pytest.fixture(scope="module")
def sentence_entropies(entropy_model):
    toks = text_to_byte_tokens("Daenerys Targaryen is in Game of Thrones")
    ent = compute_entropies_for_tokens(
        torch.tensor([toks]), entropy_model, device="cpu"
    )[0].tolist()
    return toks, ent


# --------------------------------------------------------------------------- #
# BLT §2.3 — entropy patching
# --------------------------------------------------------------------------- #


def _blt_reference_boundaries(raw, threshold):
    """BLT patcher.py: first_ids=[0,1]; entropies[1:]; +preds_truncation_len(2).

    Equivalent to: a patch starts at k iff raw[k-1] > threshold.
    """
    n = len(raw)
    starts = [0] + ([1] if n > 1 else [])
    for j, h in enumerate(raw[1:]):
        if h > threshold and j + 2 < n:
            starts.append(j + 2)
    return sorted(set(starts))


def test_boundary_uses_entropy_of_the_byte_it_starts(sentence_entropies):
    """BLT Eq. (1): H(x_i) is the entropy of p(x_i | x_<i).

    compute_entropies_for_tokens returns next-byte entropies, so H(x_k) lives at
    index k-1. Comparing index k instead shifts every boundary one byte early.
    """
    _, ent = sentence_entropies
    threshold = sorted(ent)[len(ent) // 2]
    got, _ = entropy_patch_sentence(ent, threshold)
    assert got == _blt_reference_boundaries(ent, threshold)


def test_first_two_positions_always_start_patches(sentence_entropies):
    _, ent = sentence_entropies
    boundaries, _ = entropy_patch_sentence(ent, threshold=1e9)
    assert boundaries == [0, 1]


def test_patch_lengths_tile_the_sequence(sentence_entropies):
    _, ent = sentence_entropies
    boundaries, lengths = entropy_patch_sentence(ent, sorted(ent)[len(ent) // 2])
    assert sum(lengths) == len(ent)
    assert [b + l for b, l in zip(boundaries, lengths)][:-1] == boundaries[1:]


def test_monotonicity_constraint(sentence_entropies):
    """BLT §2.3 second rule: start a patch when H(x_k) - H(x_k-1) > theta_r."""
    _, ent = sentence_entropies
    got, _ = entropy_patch_sentence(ent, 0.0, mode="monotonic", threshold_add=0.0)
    expected = [0, 1] + [
        k for k in range(2, len(ent)) if ent[k - 1] - ent[k - 2] > 0.0
    ]
    assert got == expected


def test_unknown_patching_mode_rejected(sentence_entropies):
    _, ent = sentence_entropies
    with pytest.raises(ValueError, match="global|monotonic"):
        entropy_patch_sentence(ent, 1.0, mode="nonsense")


def test_newline_context_reset_segments(entropy_model):
    """BLT §4.4 resets the entropy context at newlines."""
    toks = text_to_byte_tokens("one\ntwo\nthree")
    segments = split_on_newlines(toks)
    assert len(segments) == 3
    assert sum(len(s) for s in segments) == len(toks)
    assert all(s[-1] == ord("\n") + OFFSET for s in segments[:-1])


# --------------------------------------------------------------------------- #
# BLT §4.2 / §4.4 — entropy model context
# --------------------------------------------------------------------------- #


def test_batch_rows_do_not_leak_into_each_other(entropy_model, sentence_entropies):
    toks, ent = sentence_entropies
    both = compute_entropies_for_tokens(
        torch.tensor([toks, toks]), entropy_model, device="cpu"
    )
    assert torch.allclose(both[0], both[1], atol=1e-6)
    assert torch.allclose(both[0], torch.tensor(ent), atol=1e-6)


def test_long_sequences_keep_left_context(entropy_model):
    """Positions beyond the first window must still be scored with real context."""
    toks = [(i % 200) + OFFSET for i in range(1100)]
    full = compute_entropies_for_tokens(
        torch.tensor([toks]), entropy_model, device="cpu"
    )[0]
    assert full.shape[0] == 1100
    w = entropy_model.max_length
    probe = torch.tensor([toks[600 - w // 2 : 601]])
    ref = compute_entropies_for_tokens(probe, entropy_model, device="cpu")[0][-1]
    assert torch.allclose(full[600], ref, atol=1e-4)


def test_sliding_window_mask_shape():
    mask = sliding_window_causal_mask(6, 3, torch.device("cpu"))
    assert mask[5].tolist() == [False, False, False, True, True, True]
    assert mask[0].tolist() == [True, False, False, False, False, False]


def test_sliding_window_bounds_the_receptive_field():
    torch.manual_seed(0)
    model = ByteEntropyModel(
        dim=64, n_heads=4, n_layers=2, max_seqlen=512, attn_window=8
    ).eval()
    toks = text_to_byte_tokens("Daenerys Targaryen is in Game of Thrones")[:40]
    with torch.no_grad():
        plain = model(torch.tensor([toks]))[0, -1]
        prefixed = model(torch.tensor([[0] * 10 + toks]))[0, -1]
    assert torch.allclose(plain, prefixed, atol=1e-4)


# --------------------------------------------------------------------------- #
# BLT §3.2.1 / Appendix C — hash n-gram embeddings
# --------------------------------------------------------------------------- #


def test_rolling_poly_hash_matches_equation_23():
    """Appendix C Eq. (23): sum_j b_{i-j+1} a^{j-1}, most-recent byte exponent 0."""
    assert rolling_poly_hash([1, 2, 3]) == (
        1 * pow(HASH_BASE_PRIME, 2, MOD) + 2 * HASH_BASE_PRIME + 3
    )


def test_vectorised_hash_matches_scalar_and_skips_short_prefixes():
    tokens = torch.tensor([[5, 6, 7, 8]])
    h = _rolling_poly_hash_tensor(tokens, 3)
    assert h[0, 0].item() == -1 and h[0, 1].item() == -1
    assert h[0, 2].item() == rolling_poly_hash([5, 6, 7]) % MOD
    assert h[0, 3].item() == rolling_poly_hash([6, 7, 8]) % MOD


def test_hash_ngram_embeddings_change_the_representation():
    torch.manual_seed(0)
    emb = HashNGramEmbedder(16, ngram_sizes=(3, 4), hash_vocab_size=997)
    tokens = torch.randint(4, 260, (2, 12))
    x = torch.randn(2, 12, 16)
    out = emb(tokens, x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_hash_ngram_normalisation_is_by_num_sizes_plus_one():
    """Eq. (3) divides by the n-gram count plus one (the byte embedding)."""
    emb = HashNGramEmbedder(4, ngram_sizes=(3,), hash_vocab_size=16)
    with torch.no_grad():
        emb.tables["3"].weight.zero_()
    x = torch.ones(1, 5, 4)
    out = emb(torch.randint(4, 260, (1, 5)), x)
    assert torch.allclose(out, x / 2)


# --------------------------------------------------------------------------- #
# BLT §3.1 / §3.2 — local encoder and latent transformer
# --------------------------------------------------------------------------- #


def test_encoder_emits_a_patch_sequence_not_a_mean(sentence_entropies):
    toks, ent = sentence_entropies
    boundaries, lengths = entropy_patch_sentence(ent, sorted(ent)[len(ent) // 2])
    enc = BLTSentenceEncoder(dim=32, concept_dim=64, encoder_layers=2, latent_layers=2, n_heads=4)
    patches = enc.encoder(torch.tensor([toks]), boundaries, lengths)
    assert patches.shape == (len(boundaries), 64)


def test_whole_byte_side_receives_gradient(sentence_entropies):
    """The local encoder is trainable; BLT does not reuse the frozen entropy model."""
    toks, ent = sentence_entropies
    boundaries, lengths = entropy_patch_sentence(ent, sorted(ent)[len(ent) // 2])
    enc = BLTSentenceEncoder(dim=32, concept_dim=64, encoder_layers=2, latent_layers=2, n_heads=4)
    enc(torch.tensor([toks]), boundaries, lengths).sum().backward()
    trained = {
        name
        for name, p in enc.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    }
    assert any("hash_ngrams" in n for n in trained)
    assert any("encoder.layers" in n for n in trained)
    assert any("latent" in n for n in trained)


def test_latent_transformer_is_causal():
    torch.manual_seed(0)
    lat = BLTLatentTransformer(concept_dim=8, model_dim=8, n_layers=1, n_heads=2).eval()
    patches = torch.randn(5, 8)
    with torch.no_grad():
        base = lat(patches)
        perturbed = patches.clone()
        perturbed[4] += 10.0
        after = lat(perturbed)
    assert torch.allclose(base[:4], after[:4], atol=1e-5)


# --------------------------------------------------------------------------- #
# LCM §2.3.1 — Base-LCM
# --------------------------------------------------------------------------- #


@pytest.fixture
def small_lcm():
    torch.manual_seed(0)
    m = BaseLCM(embed_dim=16, model_dim=32, n_layers=2, n_heads=4, max_seq_len=64)
    m.fit_normalizer(torch.randn(500, 16))
    return m


def test_forward_shapes(small_lcm):
    src = torch.randn(2, 6, 16)
    assert small_lcm.forward_all(src).shape == (2, 6, 16)
    assert small_lcm(src).shape == (2, 16)
    assert small_lcm(src, torch.randn(2, 1, 16)).shape == (2, 16)
    assert small_lcm(src, torch.randn(2, 6, 16)).shape == (2, 6, 16)


def test_prefix_is_contextualised(small_lcm):
    """Decoder-only means the prefix passes through self-attention.

    The previous encoder-decoder form only linearly projected the context, so
    the model could not reason over the concept sequence at all.
    """
    small_lcm.eval()
    src = torch.randn(2, 6, 16)
    with torch.no_grad():
        base = small_lcm.forward_all(src)
        changed = src.clone()
        changed[:, 0] += 5.0
        after = small_lcm.forward_all(changed)
    assert not torch.allclose(base[:, -1], after[:, -1], atol=1e-4)


def test_base_lcm_is_causal(small_lcm):
    small_lcm.eval()
    src = torch.randn(2, 6, 16)
    with torch.no_grad():
        base = small_lcm.forward_all(src)
        changed = src.clone()
        changed[:, 5] += 5.0
        after = small_lcm.forward_all(changed)
    assert torch.allclose(base[:, 0], after[:, 0], atol=1e-5)


def test_sequence_longer_than_max_is_rejected():
    m = BaseLCM(embed_dim=8, model_dim=16, n_layers=1, n_heads=2, max_seq_len=4)
    with pytest.raises(ValueError, match="max_seq_len"):
        m.forward_all(torch.randn(1, 9, 8))


# --------------------------------------------------------------------------- #
# LCM Eqs. (1)-(4) — robust scaler
# --------------------------------------------------------------------------- #


def test_robust_scaler_round_trips():
    scaler = RobustScaler(4).fit(torch.randn(1000, 4) * 3 + 7)
    x = torch.randn(10, 4)
    assert torch.allclose(scaler.denormalize(scaler.normalize(x)), x, atol=1e-5)


def test_robust_scaler_reports_fitted_state():
    assert not RobustScaler(4).is_fitted
    assert RobustScaler(4).fit(torch.randn(50, 4)).is_fitted


def test_scaler_statistics_are_buffers_not_parameters(small_lcm):
    names = dict(small_lcm.named_buffers())
    assert "scaler.median" in names and "scaler.iqr" in names
    assert not any("scaler" in n for n in dict(small_lcm.named_parameters()))


def test_postnet_denormalizes_after_the_projection():
    """Eq. (2): outputs land back in raw coordinates, not normalized space."""
    torch.manual_seed(0)
    m = BaseLCM(embed_dim=4, model_dim=8, n_layers=1, n_heads=2, max_seq_len=8)
    m.fit_normalizer(torch.randn(500, 4) * 50 + 100)
    out = m.forward_all(torch.randn(1, 3, 4) * 50 + 100)
    assert out.abs().mean() > 5.0


# --------------------------------------------------------------------------- #
# LCM §2.3.1 — end-of-text handling
# --------------------------------------------------------------------------- #


def test_eot_is_a_buffer_not_a_learned_parameter(small_lcm):
    """It must equal encode("End of text."), so it cannot be a free Parameter."""
    assert "eot_emb" in dict(small_lcm.named_buffers())
    assert "eot_emb" not in dict(small_lcm.named_parameters())


def test_generation_refuses_to_run_without_an_eot(small_lcm):
    with pytest.raises(RuntimeError, match="EOT concept is unset"):
        small_lcm.generate_sequence(torch.randn(3, 16))


def test_generation_stops_on_eot_similarity(small_lcm):
    eot = torch.randn(16)
    small_lcm.set_eot_embedding(eot)
    small_lcm.forward = lambda src: eot.unsqueeze(0).clone()
    assert small_lcm.generate_sequence(torch.randn(3, 16), max_len=10).shape[0] == 0


def test_generation_stops_on_repetition(small_lcm):
    small_lcm.set_eot_embedding(torch.randn(16))
    repeated = torch.randn(16)
    small_lcm.forward = lambda src: repeated.unsqueeze(0).clone()
    assert small_lcm.generate_sequence(torch.randn(3, 16), max_len=10).shape[0] == 1


def test_set_eot_rejects_wrong_dimension(small_lcm):
    with pytest.raises(ValueError, match="expected 16"):
        small_lcm.set_eot_embedding(torch.randn(9))


# --------------------------------------------------------------------------- #
# LCM §2.3.2-2.3.4 — diffusion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("schedule", ["cosine", "quadratic", "sigmoid"])
def test_schedules_are_monotone_with_zero_terminal_snr(schedule):
    """Lin et al. (2024): the paper rescales every schedule to zero terminal SNR."""
    alphas_bar = make_noise_schedule(schedule, 50)
    assert alphas_bar[0] > alphas_bar[-1]
    assert torch.all(alphas_bar[:-1] >= alphas_bar[1:] - 1e-6)
    assert abs(alphas_bar[-1].item()) < 1e-6


def test_sigmoid_gamma_must_be_positive():
    with pytest.raises(ValueError, match="sigmoid_gamma"):
        make_noise_schedule("sigmoid", 20, sigmoid_gamma=-1.5)


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError, match="unknown noise schedule"):
        make_noise_schedule("triangular", 20)


def test_q_sample_is_variance_preserving():
    gd = GaussianDiffusion(50, "cosine")
    x0 = torch.randn(2048, 16)
    noise = torch.randn_like(x0)
    t = torch.full((2048,), 25, dtype=torch.long)
    xt = gd.q_sample(x0, t, noise)
    assert xt.shape == x0.shape
    assert abs(xt.std().item() - 1.0) < 0.15


def test_two_tower_trains_and_samples():
    torch.manual_seed(0)
    model = TwoTowerDiffusionLCM(
        embed_dim=8, model_dim=16, context_layers=1, denoiser_layers=2,
        n_heads=2, max_seq_len=16, timesteps=20,
    )
    model.fit_normalizer(torch.randn(200, 8))
    loss = model.loss(torch.randn(2, 5, 8))
    assert torch.isfinite(loss) and loss.dim() == 0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    model.eval()
    out = model.sample_next(torch.randn(2, 4, 8), steps=4)
    assert out.shape == (2, 8) and torch.isfinite(out).all()


def test_one_tower_trains_and_samples():
    torch.manual_seed(0)
    model = OneTowerDiffusionLCM(
        embed_dim=8, model_dim=16, n_layers=2, n_heads=2, max_seq_len=16, timesteps=20
    )
    model.fit_normalizer(torch.randn(200, 8))
    loss = model.loss(torch.randn(2, 5, 8))
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    out = model.sample_next(torch.randn(2, 3, 8), steps=3)
    assert out.shape == (2, 8) and torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# LCM §2.3.5 — quantized LCM
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fitted_rvq():
    torch.manual_seed(0)
    rvq = ResidualVectorQuantizer(dim=8, n_codebooks=3, units_per_codebook=16)
    rvq.fit(torch.randn(200, 8), iters=3, verbose=False)
    return rvq


def test_rvq_refines_coarse_to_fine(fitted_rvq):
    """Each extra codebook quantizes the previous residual, so error decreases."""
    torch.manual_seed(1)
    data = torch.randn(200, 8)
    codes = fitted_rvq.encode(data)
    assert codes.shape == (200, 3)
    errors = [
        (data - fitted_rvq.decode(codes, n)).pow(2).mean().item() for n in (1, 2, 3)
    ]
    assert errors[0] > errors[1] > errors[2]


def test_quant_lcm_requires_a_fitted_quantizer():
    model = QuantLCM(
        embed_dim=8, model_dim=16, n_layers=1, n_heads=2, max_seq_len=16,
        n_codebooks=3, units_per_codebook=16,
    )
    with pytest.raises(RuntimeError, match="fit_quantizer"):
        model.loss(torch.randn(2, 5, 8))


@pytest.mark.parametrize("target", ["discrete", "continuous"])
def test_quant_lcm_trains_and_samples(fitted_rvq, target):
    torch.manual_seed(0)
    model = QuantLCM(
        embed_dim=8, model_dim=16, n_layers=1, n_heads=2, max_seq_len=16,
        n_codebooks=3, units_per_codebook=16, target=target,
    )
    model.quantizer.load_state_dict(fitted_rvq.state_dict())
    loss = model.loss(torch.randn(2, 5, 8))
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    out = model.sample_next(torch.randn(2, 4, 8), temperature=1.0, top_k=2)
    assert out.shape == (2, 8) and torch.isfinite(out).all()


def test_quant_lcm_rejects_unknown_target():
    with pytest.raises(ValueError, match="discrete|continuous"):
        QuantLCM(embed_dim=8, target="bogus")


# --------------------------------------------------------------------------- #
# Regression: the BLTLoader API that finetune_lcm / train_base_lcm call
# --------------------------------------------------------------------------- #


def test_blt_loader_exposes_encode_sentences():
    """finetune_lcm.py and train_base_lcm.py call this; it used to not exist."""
    from blt_loader import BLTLoader

    assert hasattr(BLTLoader, "encode_sentences")
    assert hasattr(BLTLoader, "encode_sentences_batch")


def test_every_lcm_variant_is_trainable_through_one_interface():
    """train_lcm_blt.py --lcm_variant dispatches over these four models.

    Base-LCM is trained through forward_all + masked MSE; the others own their
    objective via .loss(document). Both paths must work for the CLI to be able
    to switch between them.
    """
    torch.manual_seed(0)
    doc = torch.randn(2, 5, 8)

    base = BaseLCM(embed_dim=8, model_dim=16, n_layers=1, n_heads=2, max_seq_len=16)
    base.fit_normalizer(torch.randn(200, 8))
    preds = base.forward_all(doc[:, :-1])
    assert preds.shape == (2, 4, 8)

    variants = [
        TwoTowerDiffusionLCM(
            embed_dim=8, model_dim=16, context_layers=1, denoiser_layers=1,
            n_heads=2, max_seq_len=16, timesteps=10,
        ),
        OneTowerDiffusionLCM(
            embed_dim=8, model_dim=16, n_layers=1, n_heads=2,
            max_seq_len=16, timesteps=10,
        ),
    ]
    quant = QuantLCM(
        embed_dim=8, model_dim=16, n_layers=1, n_heads=2, max_seq_len=16,
        n_codebooks=2, units_per_codebook=8,
    )
    quant.fit_quantizer(torch.randn(64, 8), iters=2, verbose=False)
    variants.append(quant)

    for model in variants:
        model.fit_normalizer(torch.randn(200, 8))
        loss = model.loss(doc)
        assert torch.isfinite(loss) and loss.dim() == 0, type(model).__name__
        loss.backward()
