from pathlib import Path


def pretrained_model_path() -> Path:
    """Return the PPO model bundled with both source and wheel installations."""

    return Path(__file__).resolve().parent / "assets" / "ppo_policy.zip"
