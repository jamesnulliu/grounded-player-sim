"""Tests for the result-store rule, mandatory W&B, and backend policy.

All CPU / stdlib / offline:
* the result store writes the documented layout and is reproducible-by-record;
* mandatory tracking *raises* when ``WANDB_API_KEY`` is absent (the rule),
  without importing or contacting wandb;
* the backend policy classifies LLM vs. non-LLM backbones and enforces
  slime+sglang for served-LLM training, exempting the board-native control;
* the trainers fail fast when the W&B key is missing, before any work.
"""

import json

import pytest

from gps.backends import (
    BackendError,
    assert_inference_backend,
    assert_llm_training_uses_slime,
    is_llm_backbone,
)
from gps.policy.board_native import BoardNativeBackbone
from gps.policy.mock_backbone import MockBackbone
from gps.policy.sglang_backbone import SGLangBackbone
from gps.results import SCHEMA_VERSION, ResultStore
from gps.tracking import TrackingError, require_wandb_key


# --- result store ------------------------------------------------------
def test_result_store_creates_documented_layout(tmp_path):
    store = ResultStore(tmp_path)
    handle = store.create("E-C2", {"lr": 1e-4, "epochs": 3})

    run_json = json.loads((handle.dir / "run.json").read_text())
    assert run_json["experiment"] == "E-C2"
    assert run_json["schema_version"] == SCHEMA_VERSION
    assert run_json["status"] == "running"
    # config + env captured up front for reproducibility.
    cfg = json.loads((handle.dir / "config.json").read_text())
    assert cfg["lr"] == pytest.approx(1e-4)
    env = json.loads((handle.dir / "env.json").read_text())
    assert "python" in env and "packages" in env


def test_result_store_run_dir_under_experiment(tmp_path):
    store = ResultStore(tmp_path)
    handle = store.create("E-C2", {})
    # <root>/<experiment-slug>/<run_id>/
    assert handle.dir.parent.name == "e-c2"
    assert handle.dir.parent.parent == tmp_path


def test_log_metrics_appends_jsonl_and_summary_merges(tmp_path):
    handle = ResultStore(tmp_path).create("E-B1", {})
    handle.log_metrics({"loss": 1.0}, step=0)
    handle.log_metrics({"loss": 0.5}, step=1)
    lines = (handle.dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["step"] == 1

    handle.set_summary({"final_nll": 0.42})
    handle.set_summary({"top1": 0.3})  # merges, not overwrites
    summary = json.loads((handle.dir / "metrics.json").read_text())
    assert summary == {"final_nll": 0.42, "top1": 0.3}


def test_attach_wandb_and_finalize_recorded(tmp_path):
    handle = ResultStore(tmp_path).create("E-C2", {})
    handle.attach_wandb(run_id="abc123", url="https://wandb.ai/x/abc123")
    handle.finalize(status="completed")
    run_json = json.loads((handle.dir / "run.json").read_text())
    assert run_json["wandb"]["run_id"] == "abc123"
    assert run_json["status"] == "completed"
    assert "finished_at" in run_json


def test_artifact_path_under_artifacts_dir(tmp_path):
    handle = ResultStore(tmp_path).create("E-C2", {})
    p = handle.artifact_path("injector.pt")
    assert p.parent.name == "artifacts"
    assert p.parent.is_dir()


# --- mandatory tracking ------------------------------------------------
def test_require_wandb_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(TrackingError):
        require_wandb_key()


def test_require_wandb_key_rejects_blank(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "   ")
    with pytest.raises(TrackingError):
        require_wandb_key()


def test_require_wandb_key_returns_key_when_set(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    assert require_wandb_key() == "deadbeef"


# --- backend policy ----------------------------------------------------
def test_is_llm_backbone_classification():
    assert is_llm_backbone(SGLangBackbone())
    assert not is_llm_backbone(BoardNativeBackbone())
    assert not is_llm_backbone(MockBackbone())


def test_board_native_training_exempt_from_slime():
    # Non-LLM backbone: no slime/sglang requirement, returns cleanly.
    assert_llm_training_uses_slime(BoardNativeBackbone(), train_backbone=True)


def test_inference_with_non_llm_backbone_is_unrestricted():
    assert_inference_backend(MockBackbone())  # no raise


def test_llm_training_requires_slime_when_absent():
    import importlib.util

    if importlib.util.find_spec("slime") is not None:  # pragma: no cover
        pytest.skip("slime installed; rule is satisfied")
    with pytest.raises(BackendError, match="slime"):
        assert_llm_training_uses_slime(SGLangBackbone(), train_backbone=False)


def test_llm_inference_requires_sglang_when_absent():
    import importlib.util

    if importlib.util.find_spec("sglang") is not None:  # pragma: no cover
        pytest.skip("sglang installed; rule is satisfied")
    with pytest.raises(BackendError, match="sglang"):
        assert_inference_backend(SGLangBackbone())


# --- trainer integration: fail fast without the W&B key ----------------
def test_sft_trainer_aborts_without_wandb_key(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    from gps.latent.structured import StructuredInjector
    from gps.train.base import TrainConfig
    from gps.train.sft import SFTTrainer

    trainer = SFTTrainer(
        StructuredInjector(),
        MockBackbone(),
        TrainConfig(results_root=str(tmp_path)),
    )
    from gps.train.base import TrajectoryDataset

    with pytest.raises(TrackingError):
        trainer.fit(TrajectoryDataset())
