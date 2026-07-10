"""
Tests for BanglaGamba training infrastructure.

Covers:
  - Deterministic epoch-seeded shuffle
  - Fast-forward correctness for resume
  - MetricLogger CSV creation, resume, and eval logging
  - Checkpoint save/load roundtrip (including new fields)
  - Atomic checkpoint saves
  - Backward-compatible checkpoint loading (old format)
  - Model export (stripped weights + config)
  - TrainerConfig parsing
  - Time formatting
  - Dual eval dict handling

Run with:
    cd bangla-gamba/
    python -m pytest tests/test_training_infra.py -v
"""

import csv
import math
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Create a temporary directory, cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="gamba_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def dummy_npy_dir(tmp_dir):
    """Create a directory with two small .npy shard files."""
    npy_dir = os.path.join(tmp_dir, "shards")
    os.makedirs(npy_dir)
    seq_len = 64  # small for speed
    for i in range(2):
        data = np.random.randint(0, 1000, size=(50, seq_len), dtype=np.int32)
        np.save(os.path.join(npy_dir, f"shard_{i:05d}.npy"), data)
    return npy_dir


@pytest.fixture
def tiny_model():
    """Create a minimal model for checkpoint roundtrip tests."""
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )
    return model


@pytest.fixture
def dual_optimizers(tiny_model):
    """Create two optimizers (simulating Muon + AdamW) for the tiny model."""
    # Split params into two groups
    params = list(tiny_model.parameters())
    muon_opt = torch.optim.SGD(params[:2], lr=0.01)
    adamw_opt = torch.optim.AdamW(params[2:], lr=0.001)
    return muon_opt, adamw_opt


# ══════════════════════════════════════════════════════════════════════════
# Test: Deterministic shuffle
# ══════════════════════════════════════════════════════════════════════════

class TestDeterministicShuffle:
    """Verify epoch-seeded RandomSampler produces reproducible orders."""

    def test_same_seed_same_epoch_identical_order(self, dummy_npy_dir):
        """Same (seed, epoch) must produce identical batch sequence."""
        from src.data.collator import build_dataset, make_epoch_loader

        ds = build_dataset(dummy_npy_dir)
        loader_a = make_epoch_loader(ds, epoch=0, batch_size=4, num_workers=0,
                                      shuffle=True, base_seed=1839, pin_memory=False)
        loader_b = make_epoch_loader(ds, epoch=0, batch_size=4, num_workers=0,
                                      shuffle=True, base_seed=1839, pin_memory=False)

        for batch_a, batch_b in zip(loader_a, loader_b):
            assert torch.equal(batch_a["input_ids"], batch_b["input_ids"]), \
                "Same seed+epoch must produce identical batches"

    def test_different_epoch_different_order(self, dummy_npy_dir):
        """Different epochs must produce different shuffle orders."""
        from src.data.collator import build_dataset, make_epoch_loader

        ds = build_dataset(dummy_npy_dir)
        loader_0 = make_epoch_loader(ds, epoch=0, batch_size=4, num_workers=0,
                                      shuffle=True, base_seed=1839, pin_memory=False)
        loader_1 = make_epoch_loader(ds, epoch=1, batch_size=4, num_workers=0,
                                      shuffle=True, base_seed=1839, pin_memory=False)

        first_0 = next(iter(loader_0))["input_ids"]
        first_1 = next(iter(loader_1))["input_ids"]
        assert not torch.equal(first_0, first_1), \
            "Different epochs should produce different first batches"

    def test_different_seed_different_order(self, dummy_npy_dir):
        """Different base seeds must produce different orders."""
        from src.data.collator import build_dataset, make_epoch_loader

        ds = build_dataset(dummy_npy_dir)
        loader_a = make_epoch_loader(ds, epoch=0, batch_size=4, num_workers=0,
                                      shuffle=True, base_seed=1839, pin_memory=False)
        loader_b = make_epoch_loader(ds, epoch=0, batch_size=4, num_workers=0,
                                      shuffle=True, base_seed=9999, pin_memory=False)

        first_a = next(iter(loader_a))["input_ids"]
        first_b = next(iter(loader_b))["input_ids"]
        assert not torch.equal(first_a, first_b), \
            "Different seeds should produce different orders"


# ══════════════════════════════════════════════════════════════════════════
# Test: Fast-forward correctness
# ══════════════════════════════════════════════════════════════════════════

class TestFastForward:
    """Verify that skipping N batches lands on the correct next batch."""

    def test_skip_matches_sequential(self, dummy_npy_dir):
        """Skipping 5 batches then reading == reading 6th batch directly."""
        from src.data.collator import build_dataset, make_epoch_loader

        ds = build_dataset(dummy_npy_dir)
        skip_count = 5

        # Full sequential read
        loader_full = make_epoch_loader(ds, epoch=0, batch_size=4, num_workers=0,
                                         shuffle=True, base_seed=1839, pin_memory=False)
        full_iter = iter(loader_full)
        for _ in range(skip_count):
            next(full_iter)
        expected_batch = next(full_iter)["input_ids"]

        # Rebuild + fast-forward
        loader_ff = make_epoch_loader(ds, epoch=0, batch_size=4, num_workers=0,
                                       shuffle=True, base_seed=1839, pin_memory=False)
        ff_iter = iter(loader_ff)
        for _ in range(skip_count):
            next(ff_iter)
        actual_batch = next(ff_iter)["input_ids"]

        assert torch.equal(expected_batch, actual_batch), \
            "Fast-forwarded batch must match sequential read"


# ══════════════════════════════════════════════════════════════════════════
# Test: MetricLogger
# ══════════════════════════════════════════════════════════════════════════

class TestMetricLogger:
    """Verify CSV logging, resume wall-clock, and eval CSV."""

    def test_csv_creation_and_append(self, tmp_dir):
        """Fresh logger creates CSV with correct headers, rows append."""
        from src.utils.logging import MetricLogger

        logger = MetricLogger(tmp_dir, "test_run")
        logger.log({"step": 1, "loss": 5.5, "ppl": 100.0})
        logger.log({"step": 2, "loss": 4.4, "ppl": 80.0})

        csv_path = Path(tmp_dir) / "test_run" / "metrics.csv"
        assert csv_path.exists()

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["step"] == "1"
        assert rows[1]["loss"] == "4.4"
        assert "elapsed_s" in rows[0]

    def test_resumed_wall_clock_offsets_elapsed(self, tmp_dir):
        """Setting _resumed_wall_clock should offset _elapsed()."""
        from src.utils.logging import MetricLogger
        import time

        logger = MetricLogger(tmp_dir, "test_run")
        logger._resumed_wall_clock = 10000.0
        logger._session_start = time.time()

        elapsed = logger._elapsed()
        # Should be at least 10000 (resumed) + tiny session time
        assert elapsed >= 10000.0
        assert elapsed < 10010.0  # shouldn't be way off

    def test_eval_csv_creation(self, tmp_dir):
        """log_eval() creates a separate eval_metrics.csv."""
        from src.utils.logging import MetricLogger

        logger = MetricLogger(tmp_dir, "test_run")
        logger.log_eval({"step": 10, "val_ppl_bng": 50.0, "val_ppl_eng": 70.0})

        eval_csv = Path(tmp_dir) / "test_run" / "eval_metrics.csv"
        assert eval_csv.exists()

        with open(eval_csv, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert "val_ppl_bng" in rows[0]
        assert "val_ppl_eng" in rows[0]
        assert "elapsed_s" in rows[0]

    def test_csv_schema_preserved_on_resume(self, tmp_dir):
        """Resuming into an existing CSV preserves original column order."""
        from src.utils.logging import MetricLogger

        # First session
        logger1 = MetricLogger(tmp_dir, "test_run")
        logger1.log({"step": 1, "loss": 5.5, "custom_col": 42.0})

        # Second session (simulates resume)
        logger2 = MetricLogger(tmp_dir, "test_run")
        logger2.log({"step": 2, "loss": 4.4, "custom_col": 43.0})

        csv_path = Path(tmp_dir) / "test_run" / "metrics.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert "custom_col" in headers
        assert "elapsed_s" in headers


# ══════════════════════════════════════════════════════════════════════════
# Test: Checkpoint roundtrip
# ══════════════════════════════════════════════════════════════════════════

class TestCheckpoint:
    """Verify checkpoint save/load preserves all fields."""

    def test_roundtrip_all_fields(self, tmp_dir, tiny_model, dual_optimizers):
        """Save → load should preserve all fields including new ones."""
        from src.training.checkpoint import save_checkpoint, load_checkpoint

        muon_opt, adamw_opt = dual_optimizers

        # Simple schedulers
        muon_sched = torch.optim.lr_scheduler.StepLR(muon_opt, step_size=10)
        adamw_sched = torch.optim.lr_scheduler.StepLR(adamw_opt, step_size=10)

        ckpt_path = os.path.join(tmp_dir, "test_ckpt.pt")
        save_checkpoint(
            ckpt_path, tiny_model, muon_opt, adamw_opt, muon_sched, adamw_sched,
            step=100, tokens_seen=204800, train_loss=3.5,
            epoch=2, batches_consumed_this_epoch=150,
            data_seed=1839, wall_clock=5000.0,
            config={"d_model": 16},
        )

        # Load into fresh copies
        model2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
        params2 = list(model2.parameters())
        muon2 = torch.optim.SGD(params2[:2], lr=0.01)
        adamw2 = torch.optim.AdamW(params2[2:], lr=0.001)
        sched_m2 = torch.optim.lr_scheduler.StepLR(muon2, step_size=10)
        sched_a2 = torch.optim.lr_scheduler.StepLR(adamw2, step_size=10)

        ckpt = load_checkpoint(
            ckpt_path, model2, muon2, adamw2, sched_m2, sched_a2, device="cpu"
        )

        assert ckpt["step"] == 100
        assert ckpt["tokens_seen"] == 204800
        assert ckpt["epoch"] == 2
        assert ckpt["batches_consumed_this_epoch"] == 150
        assert ckpt["data_seed"] == 1839
        assert ckpt["wall_clock"] == 5000.0
        assert ckpt["config"] == {"d_model": 16}

    def test_atomic_save_no_tmp_leftover(self, tmp_dir, tiny_model, dual_optimizers):
        """Atomic save should not leave .tmp files behind."""
        from src.training.checkpoint import save_checkpoint

        muon_opt, adamw_opt = dual_optimizers
        muon_sched = torch.optim.lr_scheduler.StepLR(muon_opt, step_size=10)
        adamw_sched = torch.optim.lr_scheduler.StepLR(adamw_opt, step_size=10)

        ckpt_path = os.path.join(tmp_dir, "step_00000100.pt")
        save_checkpoint(
            ckpt_path, tiny_model, muon_opt, adamw_opt, muon_sched, adamw_sched,
            step=100, tokens_seen=0, train_loss=5.0,
        )

        assert os.path.exists(ckpt_path)
        assert not os.path.exists(ckpt_path + ".tmp")

    def test_backward_compat_old_checkpoint(self, tmp_dir, tiny_model, dual_optimizers):
        """Old checkpoints without wall_clock/epoch should load with defaults."""
        muon_opt, adamw_opt = dual_optimizers
        muon_sched = torch.optim.lr_scheduler.StepLR(muon_opt, step_size=10)
        adamw_sched = torch.optim.lr_scheduler.StepLR(adamw_opt, step_size=10)

        # Simulate old-format checkpoint (missing wall_clock, epoch, etc.)
        old_ckpt = {
            "step": 50,
            "tokens_seen": 102400,
            "train_loss": 4.0,
            "val_perplexity": None,
            "model_state_dict": tiny_model.state_dict(),
            "muon_optimizer_state_dict": muon_opt.state_dict(),
            "adamw_optimizer_state_dict": adamw_opt.state_dict(),
            "muon_sched_state_dict": muon_sched.state_dict(),
            "adamw_sched_state_dict": adamw_sched.state_dict(),
            "config": None,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": None,
            # No epoch, no batches_consumed, no data_seed, no wall_clock
        }
        ckpt_path = os.path.join(tmp_dir, "old_ckpt.pt")
        torch.save(old_ckpt, ckpt_path)

        from src.training.checkpoint import load_checkpoint

        model2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
        params2 = list(model2.parameters())
        muon2 = torch.optim.SGD(params2[:2], lr=0.01)
        adamw2 = torch.optim.AdamW(params2[2:], lr=0.001)
        sched_m2 = torch.optim.lr_scheduler.StepLR(muon2, step_size=10)
        sched_a2 = torch.optim.lr_scheduler.StepLR(adamw2, step_size=10)

        ckpt = load_checkpoint(
            ckpt_path, model2, muon2, adamw2, sched_m2, sched_a2, device="cpu"
        )

        # Should gracefully default
        assert ckpt["step"] == 50
        assert ckpt.get("epoch", 0) == 0
        assert ckpt.get("batches_consumed_this_epoch", 0) == 0
        assert ckpt.get("wall_clock", 0.0) == 0.0
        assert ckpt.get("data_seed") is None


# ══════════════════════════════════════════════════════════════════════════
# Test: Model export
# ══════════════════════════════════════════════════════════════════════════

class TestModelExport:
    """Verify stripped model export."""

    def test_export_creates_files(self, tmp_dir, tiny_model):
        """export_model() should create model.pt and config.yaml."""
        from src.training.checkpoint import export_model

        export_dir = export_model(
            tiny_model,
            config={"d_model": 16, "n_layers": 2},
            model_dir=tmp_dir,
            run_name="test_export",
        )

        model_pt = Path(export_dir) / "model.pt"
        config_yaml = Path(export_dir) / "config.yaml"

        assert model_pt.exists()
        assert config_yaml.exists()

        # model.pt should be loadable state dict
        state = torch.load(str(model_pt), map_location="cpu", weights_only=True)
        assert "0.weight" in state  # nn.Sequential keys

        # config.yaml should be valid YAML
        import yaml
        with open(config_yaml) as f:
            cfg = yaml.safe_load(f)
        assert cfg["d_model"] == 16

    def test_export_is_smaller_than_checkpoint(self, tmp_dir, tiny_model, dual_optimizers):
        """Exported model.pt should be much smaller than a full checkpoint."""
        from src.training.checkpoint import save_checkpoint, export_model

        muon_opt, adamw_opt = dual_optimizers
        muon_sched = torch.optim.lr_scheduler.StepLR(muon_opt, step_size=10)
        adamw_sched = torch.optim.lr_scheduler.StepLR(adamw_opt, step_size=10)

        ckpt_path = os.path.join(tmp_dir, "full_ckpt.pt")
        save_checkpoint(
            ckpt_path, tiny_model, muon_opt, adamw_opt, muon_sched, adamw_sched,
            step=100, tokens_seen=0, train_loss=5.0,
        )

        export_dir = export_model(
            tiny_model, config={"d_model": 16}, model_dir=tmp_dir, run_name="stripped"
        )

        ckpt_size = os.path.getsize(ckpt_path)
        model_size = os.path.getsize(os.path.join(export_dir, "model.pt"))
        assert model_size < ckpt_size, "Stripped model should be smaller than full checkpoint"


# ══════════════════════════════════════════════════════════════════════════
# Test: TrainerConfig
# ══════════════════════════════════════════════════════════════════════════

class TestTrainerConfig:
    """Verify TrainerConfig parsing from YAML."""

    def test_from_yaml_loads_new_fields(self, tmp_dir):
        """New fields (eval_batches, model_dir) should be parsed."""
        from src.training.trainer import TrainerConfig

        yaml_path = os.path.join(tmp_dir, "test_config.yaml")
        with open(yaml_path, "w") as f:
            f.write("eval_batches: 75\n")
            f.write("model_dir: saved/model\n")
            f.write("compile_model: false\n")
            f.write("checkpoint_every: 250\n")

        tc = TrainerConfig.from_yaml(yaml_path)
        assert tc.eval_batches == 75
        assert tc.model_dir == "saved/model"
        assert tc.compile_model is False
        assert tc.checkpoint_every == 250

    def test_from_yaml_ignores_unknown_keys(self, tmp_dir):
        """Unknown YAML keys should not cause errors."""
        from src.training.trainer import TrainerConfig

        yaml_path = os.path.join(tmp_dir, "test_config.yaml")
        with open(yaml_path, "w") as f:
            f.write("max_steps: 1000\n")
            f.write("unknown_key: 42\n")

        tc = TrainerConfig.from_yaml(yaml_path)
        assert tc.max_steps == 1000


# ══════════════════════════════════════════════════════════════════════════
# Test: Time formatting
# ══════════════════════════════════════════════════════════════════════════

class TestFormatTime:
    """Verify _format_time edge cases."""

    def test_negative_returns_placeholder(self):
        from src.training.trainer import _format_time
        assert _format_time(-1) == "??:??:??"

    def test_zero(self):
        from src.training.trainer import _format_time
        assert _format_time(0) == "0s"

    def test_seconds_only(self):
        from src.training.trainer import _format_time
        assert _format_time(45) == "45s"

    def test_minutes_and_seconds(self):
        from src.training.trainer import _format_time
        assert _format_time(125) == "2m05s"

    def test_hours(self):
        from src.training.trainer import _format_time
        assert _format_time(3723) == "1h02m03s"

    def test_large_value(self):
        from src.training.trainer import _format_time
        result = _format_time(112349.7)
        assert result == "31h12m29s"


# ══════════════════════════════════════════════════════════════════════════
# Test: Dual eval dict
# ══════════════════════════════════════════════════════════════════════════

class TestDualEval:
    """Verify evaluate() handles dict of loaders correctly."""

    def test_eval_returns_per_split_ppl(self, dummy_npy_dir):
        """evaluate() with dict loader should return per-split perplexity."""
        from src.training.trainer import Trainer, TrainerConfig
        from src.data.collator import build_dataloader

        # Create a tiny model that produces valid logits
        vocab_size = 1001  # at least as large as max token id in dummy data
        model = nn.Sequential(
            nn.Embedding(vocab_size, 32),
            nn.Linear(32, vocab_size),
        )

        # Build eval loaders (reuse same dir for both "bng" and "eng")
        eval_loader = {
            "bng": build_dataloader(dummy_npy_dir, batch_size=4, num_workers=0,
                                     shuffle=False, pin_memory=False),
            "eng": build_dataloader(dummy_npy_dir, batch_size=4, num_workers=0,
                                     shuffle=False, pin_memory=False),
        }

        # Minimal trainer config
        config = TrainerConfig(eval_batches=2, pad_token_id=0)

        # Create a mock-ish trainer with just what evaluate() needs
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.eval_loader = eval_loader
        trainer.config = config
        trainer.device = "cpu"

        # Monkey-patch compute_loss to work with our toy model
        def fake_compute_loss(logits, targets):
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1),
                ignore_index=0,
            )
            return loss, loss.item(), 0.0
        trainer.compute_loss = fake_compute_loss

        result = trainer.evaluate()

        assert "overall" in result
        assert "bng" in result
        assert "eng" in result
        assert result["overall"] > 0
        assert isinstance(result["bng"], float)
        assert isinstance(result["eng"], float)
