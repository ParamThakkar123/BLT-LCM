from unittest.mock import Mock

import pytest
import torch

from lcm_scripts import finetune_lcm
from lcm_scripts.data_loader import EmbeddingDataset, collate_embeddings


def parse_args(*extra_args):
    return finetune_lcm.build_arg_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--entropy_model",
            "entropy.pt",
            *extra_args,
        ]
    )


def test_arg_parser_accepts_device_and_lora_options():
    args = parse_args(
        "--device",
        "cpu",
        "--fraction",
        "0.25",
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--lora",
        "--target_modules",
        "linear",
        "proj",
    )

    assert args.device == "cpu"
    assert args.fraction == 0.25
    assert args.epochs == 1
    assert args.batch_size == 2
    assert args.lora is True
    assert args.qlora is False
    assert args.target_modules == ["linear", "proj"]


def test_embedding_dataset_filters_short_sequences_and_collates():
    sequences = [torch.ones(1, 4), torch.ones(2, 4), torch.ones(4, 4)]

    dataset = EmbeddingDataset(sequences)
    assert len(dataset) == 2

    src, tgt = collate_embeddings([dataset[0], dataset[1]])

    assert src.shape == (2, 3, 4)
    assert tgt.shape == (2, 3, 4)
    assert torch.equal(src[0, 0], sequences[1][0])
    assert torch.equal(tgt[0, 0], sequences[1][1])
    assert torch.equal(src[1, :3], sequences[2][:-1])
    assert torch.equal(tgt[1, :3], sequences[2][1:])


def test_configure_peft_model_noops_when_lora_disabled():
    model = torch.nn.Linear(2, 2)
    args = parse_args()
    load_components = Mock()

    result = finetune_lcm.configure_peft_model(model, args, load_components)

    assert result is model
    load_components.assert_not_called()


def test_configure_peft_model_applies_lora_with_lazy_components():
    model = torch.nn.Linear(2, 2)
    args = parse_args("--lora", "--lora_rank", "4", "--lora_alpha", "16")
    calls = []

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(("config", kwargs))

    def fake_get_peft_model(model_arg, config):
        calls.append(("get_peft_model", model_arg, config.kwargs))
        return "peft-model"

    def fake_prepare_model_for_kbit_training(model_arg, **kwargs):
        calls.append(("prepare", model_arg, kwargs))
        return model_arg

    result = finetune_lcm.configure_peft_model(
        model,
        args,
        load_components=lambda: (
            FakeLoraConfig,
            fake_get_peft_model,
            fake_prepare_model_for_kbit_training,
        ),
    )

    assert result == "peft-model"
    assert calls == [
        (
            "config",
            {
                "r": 4,
                "lora_alpha": 16,
                "target_modules": ["linear"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "SEQ_2_SEQ_LM",
            },
        ),
        (
            "get_peft_model",
            model,
            {
                "r": 4,
                "lora_alpha": 16,
                "target_modules": ["linear"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "SEQ_2_SEQ_LM",
            },
        ),
    ]


def test_configure_peft_model_prepares_qlora_before_lora():
    model = torch.nn.Linear(2, 2)
    prepared_model = torch.nn.Linear(2, 2)
    args = parse_args("--qlora")
    calls = []

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            calls.append(("config", kwargs))

    def fake_get_peft_model(model_arg, config):
        calls.append(("get_peft_model", model_arg))
        return "qlora-model"

    def fake_prepare_model_for_kbit_training(model_arg, **kwargs):
        calls.append(("prepare", model_arg, kwargs))
        return prepared_model

    result = finetune_lcm.configure_peft_model(
        model,
        args,
        load_components=lambda: (
            FakeLoraConfig,
            fake_get_peft_model,
            fake_prepare_model_for_kbit_training,
        ),
    )

    assert result == "qlora-model"
    assert calls[0] == ("prepare", model, {"use_gradient_checkpointing": False})
    assert calls[1][0] == "config"
    assert calls[2] == ("get_peft_model", prepared_model)


def test_load_peft_components_raises_clear_error_when_dependency_missing(monkeypatch):
    monkeypatch.setattr(finetune_lcm, "PEFT_AVAILABLE", False)

    with pytest.raises(ImportError, match="PEFT is required for --lora/--qlora"):
        finetune_lcm.load_peft_components()


def test_imports_expose_testable_helpers():
    assert callable(finetune_lcm.prepare_data)
    assert callable(finetune_lcm.build_arg_parser)
    assert callable(finetune_lcm.configure_peft_model)
    assert callable(finetune_lcm.EmbeddingDataset)
    assert callable(finetune_lcm.collate)
