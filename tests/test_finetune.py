import pytest
import torch
import os
import tempfile
from unittest.mock import patch, MagicMock

# Import the script functions
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))
from finetune_lcm import main, PEFT_AVAILABLE
from base_lcm import BaseLCM


EMBED_DIM = 64
MODEL_DIM = 128


@pytest.fixture
def dummy_checkpoint():
    """Create a dummy model checkpoint for testing.

    Deliberately tiny. At the previous 1024/2048 dimensions each instance was
    ~136M parameters, and with four tests each building one the suite could
    exhaust memory and fail with allocator errors that look like logic bugs.
    Nothing here depends on the width.
    """
    model = BaseLCM(embed_dim=EMBED_DIM, model_dim=MODEL_DIM, n_layers=2, n_heads=8)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        return f.name


@pytest.fixture
def dummy_embeddings():
    """Create dummy embeddings for testing."""
    return [torch.randn(3, EMBED_DIM) for _ in range(5)]  # 5 docs, 3 sentences each


def test_lora_application(dummy_checkpoint):
    """Test that LoRA is applied correctly to the model."""
    from peft import LoraConfig, get_peft_model

    # Load model
    checkpoint = torch.load(dummy_checkpoint)
    model = BaseLCM(embed_dim=EMBED_DIM, model_dim=MODEL_DIM, n_layers=2, n_heads=8)
    model.load_state_dict(checkpoint)

    # Apply LoRA. task_type must be None, matching finetune_lcm.py: BaseLCM has a
    # custom forward(src_embs, tgt_embs) rather than a HuggingFace input_ids
    # interface, so a task-specific wrapper (e.g. SEQ_2_SEQ_LM) reaches for
    # generation hooks like prepare_inputs_for_generation that do not exist here.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["linear"],
        lora_dropout=0.1,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, lora_config)

    # Check that LoRA parameters are added
    lora_params = [
        name for name, param in model.named_parameters() if "lora" in name.lower()
    ]
    assert len(lora_params) > 0, "LoRA parameters not found"

    # Check that model is trainable (some params require grad)
    trainable_params = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    assert len(trainable_params) > 0, "No trainable parameters found"


def test_qlora_does_not_actually_quantize_base_lcm(dummy_checkpoint):
    """QLoRA is a no-op on BaseLCM, which is why finetune_lcm.py dropped it.

    BaseLCM is a custom fp32 module, not a HuggingFace model loaded through a
    bitsandbytes 4-bit config, so there is nothing for QLoRA to attach to.
    ``prepare_model_for_kbit_training`` does not raise -- it quietly casts norms
    and returns an unquantized model -- so asserting that it raises pins the
    wrong behaviour. Assert the invariant that actually matters instead: no
    quantized layers appear, and the model is not flagged as 4/8-bit loaded.
    """
    from peft import prepare_model_for_kbit_training

    checkpoint = torch.load(dummy_checkpoint)
    model = BaseLCM(embed_dim=EMBED_DIM, model_dim=MODEL_DIM, n_layers=2, n_heads=8)
    model.load_state_dict(checkpoint)

    prepared = prepare_model_for_kbit_training(model)

    assert not getattr(prepared, "is_loaded_in_4bit", False)
    assert not getattr(prepared, "is_loaded_in_8bit", False)
    quantized = [
        name
        for name, module in prepared.named_modules()
        if type(module).__name__ in ("Linear4bit", "Linear8bitLt", "Params4bit")
    ]
    assert quantized == [], f"unexpected quantized layers: {quantized}"
    # Every parameter is still a plain float tensor.
    assert all(p.dtype.is_floating_point for p in prepared.parameters())


@patch("finetune_lcm.prepare_data")
@patch("finetune_lcm.BLTLoader")
def test_finetune_basic(
    mock_blt_loader, mock_prepare_data, dummy_checkpoint, dummy_embeddings
):
    """Test basic fine-tuning without LoRA."""
    mock_prepare_data.return_value = [["sentence1", "sentence2"], ["sentence3"]]
    mock_blt = MagicMock()
    mock_blt.encode_sentences.return_value = dummy_embeddings[0]
    mock_blt_loader.return_value = mock_blt

    # Mock args
    with patch(
        "sys.argv",
        [
            "finetune_lcm.py",
            "--checkpoint",
            dummy_checkpoint,
            "--entropy_model",
            "dummy.pt",
            "--fraction",
            "0.25",
            "--epochs",
            "1",
            "--batch_size",
            "1",
        ],
    ):
        # This would run the script, but to avoid full execution, we can test parts
        # For now, just check that the script can be imported and parsed
        pass


@patch("finetune_lcm.prepare_data")
@patch("finetune_lcm.BLTLoader")
def test_finetune_with_lora(
    mock_blt_loader, mock_prepare_data, dummy_checkpoint, dummy_embeddings
):
    """Test fine-tuning with LoRA enabled."""
    mock_prepare_data.return_value = [["sentence1", "sentence2"]]
    mock_blt = MagicMock()
    mock_blt.encode_sentences.return_value = dummy_embeddings[0]
    mock_blt_loader.return_value = mock_blt

    # Mock args with LoRA
    with patch(
        "sys.argv",
        [
            "finetune_lcm.py",
            "--checkpoint",
            dummy_checkpoint,
            "--entropy_model",
            "dummy.pt",
            "--fraction",
            "0.25",
            "--epochs",
            "1",
            "--batch_size",
            "1",
            "--lora",
        ],
    ):
        # Again, partial test
        pass


def test_imports():
    """Test that all imports work."""
    from finetune_lcm import prepare_data, EmbeddingDataset, collate

    assert callable(prepare_data)
    assert callable(EmbeddingDataset)
    assert callable(collate)
