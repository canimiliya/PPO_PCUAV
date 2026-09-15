import argparse
import json
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from .assets import pretrained_model_path
from .training_environment import CurriculumPayloadUAVEnv


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    total_timesteps: int = 20_000_000
    n_envs: int = 6
    seed: int = 0
    start_level: int = 1
    resume: Path | None = None
    device: str = "auto"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    entropy_coefficient: float = 0.02
    checkpoint_frequency: int = 200_000
    evaluation_frequency: int = 50_000
    evaluation_episodes: int = 12
    tensorboard: bool = False
    start_phase: str = "auto"
    finetune_level_26: bool = False
    wire_to_ground_probability: float = 0.85
    smoke_test: bool = False


def required_success_rate(level: int) -> float:
    if level <= 15:
        return 0.80
    if level <= 19:
        return 0.95
    if level <= 24:
        return 0.75
    if level == 25:
        return 0.85
    return 0.95


def _set_task_phase(vector_env, n_envs: int, phase: str) -> None:
    if phase == "wire_only":
        vector_env.env_method("set_fixed_task_mode", "wire_to_ground")
    elif phase == "mixed":
        ground_to_wire = list(range(0, n_envs, 2))
        wire_to_ground = list(range(1, n_envs, 2))
        if ground_to_wire:
            vector_env.env_method(
                "set_fixed_task_mode", "ground_to_wire", indices=ground_to_wire
            )
        if wire_to_ground:
            vector_env.env_method(
                "set_fixed_task_mode", "wire_to_ground", indices=wire_to_ground
            )
    elif phase == "none":
        vector_env.env_method("set_fixed_task_mode", None)
    else:
        raise ValueError(f"Unknown task phase: {phase}")


class CurriculumCallback(BaseCallback):
    """Advance through 26 levels using rolling episode success rates."""

    def __init__(
        self,
        *,
        start_level: int,
        completed_models_dir: Path,
        state_path: Path,
        evaluation_env=None,
        check_frequency: int = 10_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.current_level = int(np.clip(start_level, 1, 26))
        self.completed_models_dir = Path(completed_models_dir)
        self.state_path = Path(state_path)
        self.evaluation_env = evaluation_env
        self.check_frequency = int(check_frequency)
        self.successes: deque[bool] = deque(maxlen=200)
        self._previous_wire_only: bool | None = None

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_level", self.current_level)
        if self.evaluation_env is not None:
            self.evaluation_env.env_method("set_level", self.current_level)
        self._save_state()
        if self.verbose:
            print(f"[Curriculum] start at level {self.current_level}")

    def _save_state(self) -> None:
        self.state_path.write_text(
            json.dumps({"current_level": self.current_level}, indent=2),
            encoding="utf-8",
        )

    def _all_environments_wire_only(self) -> bool:
        modes = self.training_env.get_attr("fixed_task_mode")
        return bool(modes) and all(mode == "wire_to_ground" for mode in modes)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done:
                self.successes.append(bool((info or {}).get("is_success", False)))

        if self.n_calls % self.check_frequency != 0 or len(self.successes) <= 50:
            return True

        success_rate = float(np.mean(self.successes))
        wire_only = self.current_level >= 20 and self._all_environments_wire_only()
        if self._previous_wire_only is True and not wire_only:
            self.successes.clear()
            if self.verbose:
                print(f"[Curriculum] L{self.current_level}: mixed phase buffer reset")
            self._previous_wire_only = wire_only
            return True
        self._previous_wire_only = wire_only
        threshold = required_success_rate(self.current_level)
        if self.verbose:
            print(
                f"[Curriculum] steps={self.num_timesteps} level={self.current_level} "
                f"success={success_rate:.2f}/{threshold:.2f}"
            )
        if success_rate < threshold or self.current_level >= 26 or wire_only:
            return True

        completed_level = self.current_level
        self.current_level += 1
        self.completed_models_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self.completed_models_dir / f"level_{completed_level:02d}_complete")
        self.successes.clear()
        self.training_env.env_method("set_level", self.current_level)
        if self.evaluation_env is not None:
            self.evaluation_env.env_method("set_level", self.current_level)
        self._save_state()
        if self.verbose:
            print(f"[Curriculum] upgraded to level {self.current_level}")
        return True


class TaskModePhaseCallback(BaseCallback):
    """For L20+, learn wire-to-ground first, then switch to a 50/50 mixture."""

    def __init__(
        self,
        *,
        n_envs: int,
        start_phase: str = "auto",
        check_frequency: int = 10_000,
        threshold: float = 0.80,
        minimum_samples: int = 80,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.n_envs = int(n_envs)
        self.start_phase = start_phase
        self.check_frequency = int(check_frequency)
        self.threshold = float(threshold)
        self.minimum_samples = int(minimum_samples)
        self.phase = "none"
        self.level: int | None = None
        self.wire_to_ground_successes: deque[bool] = deque(maxlen=200)

    def _apply_level_phase(self, level: int) -> None:
        self.level = level
        self.wire_to_ground_successes.clear()
        if level < 20 or self.start_phase == "none":
            self.phase = "none"
        elif self.start_phase == "mixed":
            self.phase = "mixed"
        else:
            self.phase = "wire_only"
        _set_task_phase(self.training_env, self.n_envs, self.phase)
        if self.verbose:
            print(f"[ModePhase] L{level}: {self.phase}")

    def _on_training_start(self) -> None:
        level = int(self.training_env.get_attr("level")[0])
        self._apply_level_phase(level)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if not done:
                continue
            info = info or {}
            level = int(info.get("level", self.level or 1))
            if level != self.level:
                self._apply_level_phase(level)
            if info.get("task_mode") == "wire_to_ground":
                self.wire_to_ground_successes.append(bool(info.get("is_success", False)))

        if self.phase != "wire_only" or self.n_calls % self.check_frequency != 0:
            return True
        if len(self.wire_to_ground_successes) < self.minimum_samples:
            return True
        success_rate = float(np.mean(self.wire_to_ground_successes))
        if self.verbose:
            print(
                f"[ModePhase] wire_to_ground success={success_rate:.2f} "
                f"(n={len(self.wire_to_ground_successes)})"
            )
        if success_rate >= self.threshold:
            self.phase = "mixed"
            self.wire_to_ground_successes.clear()
            _set_task_phase(self.training_env, self.n_envs, "mixed")
            if self.verbose:
                print("[ModePhase] switched to mixed 50/50 sampling")
        return True


def _environment_factory(rank: int, seed: int, level: int):
    def initialize():
        environment = CurriculumPayloadUAVEnv()
        environment.set_level(level)
        environment.reset(seed=seed + rank)
        return environment

    return initialize


def _make_training_env(config: TrainingConfig):
    factories = [
        _environment_factory(rank, config.seed, config.start_level)
        for rank in range(config.n_envs)
    ]
    vector_env = (
        DummyVecEnv(factories)
        if config.n_envs == 1
        else SubprocVecEnv(factories, start_method="spawn")
    )
    return VecMonitor(vector_env)


def _make_evaluation_env(config: TrainingConfig):
    environment = DummyVecEnv(
        [_environment_factory(0, config.seed + 100_000, config.start_level)]
    )
    return VecMonitor(environment)


def _latest_resume_model(output_dir: Path) -> Path | None:
    candidates = list((output_dir / "checkpoints").glob("*.zip"))
    candidates.extend(
        path
        for path in (output_dir / "interrupted_model.zip", output_dir / "final_model.zip")
        if path.exists()
    )
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_resume_path(value: str | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    if value == "pretrained":
        path = pretrained_model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Bundled pretrained model not found: {path}")
        return path
    if value == "auto":
        path = _latest_resume_model(output_dir)
        if path is None:
            raise FileNotFoundError(f"No checkpoint found under {output_dir}")
        return path
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume model not found: {path}")
    return path


def infer_start_level(output_dir: Path) -> int:
    state_path = output_dir / "training_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            return int(np.clip(int(state["current_level"]), 1, 26))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    completed = list((output_dir / "completed_levels").glob("level_*_complete.zip"))
    levels = []
    for path in completed:
        try:
            levels.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return min(max(levels, default=0) + 1, 26)


def _new_model(config: TrainingConfig, environment) -> PPO:
    return PPO(
        "MlpPolicy",
        environment,
        verbose=1,
        tensorboard_log=(
            str(config.output_dir / "tensorboard") if config.tensorboard else None
        ),
        device=config.device,
        seed=config.seed,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=config.entropy_coefficient,
        max_grad_norm=0.5,
        policy_kwargs={
            "activation_fn": nn.Tanh,
            "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
        },
    )


def train(config: TrainingConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    training_env = _make_training_env(config)
    evaluation_env = _make_evaluation_env(config)
    model: PPO | None = None
    try:
        if config.resume is None:
            model = _new_model(config, training_env)
        else:
            model = PPO.load(
                config.resume,
                env=training_env,
                device=config.device,
                tensorboard_log=(
                    str(config.output_dir / "tensorboard")
                    if config.tensorboard
                    else None
                ),
            )
            model.ent_coef = config.entropy_coefficient
            model.learning_rate = config.learning_rate
            model.lr_schedule = lambda _: config.learning_rate
            model.set_random_seed(config.seed)
            print(f"[Resume] {config.resume}")

        callbacks: list[BaseCallback] = []
        if config.checkpoint_frequency > 0:
            checkpoint_dir = config.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(config.checkpoint_frequency // config.n_envs, 1),
                    save_path=str(checkpoint_dir),
                    name_prefix="ppo_pcuav",
                )
            )
        if config.evaluation_frequency > 0:
            callbacks.append(
                EvalCallback(
                    evaluation_env,
                    best_model_save_path=str(config.output_dir),
                    log_path=str(config.output_dir),
                    eval_freq=max(config.evaluation_frequency // config.n_envs, 1),
                    n_eval_episodes=config.evaluation_episodes,
                    deterministic=True,
                    render=False,
                )
            )

        if config.finetune_level_26:
            training_env.env_method("set_level", 26)
            training_env.env_method("set_fixed_task_mode", None)
            training_env.env_method(
                "set_task_mode_probability", config.wire_to_ground_probability
            )
            evaluation_env.env_method("set_level", 26)
        else:
            callbacks.extend(
                [
                    CurriculumCallback(
                        start_level=config.start_level,
                        completed_models_dir=config.output_dir / "completed_levels",
                        state_path=config.output_dir / "training_state.json",
                        evaluation_env=evaluation_env,
                    ),
                    TaskModePhaseCallback(
                        n_envs=config.n_envs, start_phase=config.start_phase
                    ),
                ]
            )

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=config.resume is None,
        )
        output_model = config.output_dir / (
            "smoke_model" if config.smoke_test else "final_model"
        )
        model.save(output_model)
        return output_model.with_suffix(".zip")
    except KeyboardInterrupt:
        if model is not None:
            model.save(config.output_dir / "interrupted_model")
        raise
    except Exception:
        if model is not None:
            model.save(config.output_dir / "crash_model")
        print(traceback.format_exc())
        raise
    finally:
        training_env.close()
        evaluation_env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the hierarchical PPO-PID policy with the 26-level curriculum."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training_runs/default"))
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-level", type=int, choices=range(1, 27), default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Model zip path, 'pretrained' for the bundled policy, or 'auto' "
            "for the newest checkpoint in output-dir."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--checkpoint-frequency", type=int, default=200_000)
    parser.add_argument("--evaluation-frequency", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=12)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument(
        "--start-phase",
        choices=("auto", "mixed", "wire_only", "none"),
        default="auto",
    )
    parser.add_argument("--finetune-level-26", action="store_true")
    parser.add_argument("--wire-to-ground-probability", type=float, default=0.85)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny one-environment training job to verify the pipeline.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    output_dir = args.output_dir.resolve()
    smoke = bool(args.smoke_test)
    resume = resolve_resume_path(args.resume, output_dir)
    if args.finetune_level_26:
        start_level = 26
    elif args.start_level is not None:
        start_level = int(args.start_level)
    elif resume is not None:
        start_level = infer_start_level(output_dir)
    else:
        start_level = 1
    total_timesteps = (
        128
        if smoke
        else int(args.total_timesteps)
        if args.total_timesteps is not None
        else 8_000_000
        if args.finetune_level_26
        else 20_000_000
    )
    entropy_coefficient = (
        float(args.entropy_coefficient)
        if args.entropy_coefficient is not None
        else 0.04
        if args.finetune_level_26
        else 0.02
    )
    return TrainingConfig(
        output_dir=output_dir,
        total_timesteps=total_timesteps,
        n_envs=1 if smoke else int(args.n_envs),
        seed=int(args.seed),
        start_level=start_level,
        resume=resume,
        device=args.device,
        learning_rate=float(args.learning_rate),
        n_steps=64 if smoke else int(args.n_steps),
        batch_size=64 if smoke else int(args.batch_size),
        n_epochs=1 if smoke else int(args.n_epochs),
        entropy_coefficient=entropy_coefficient,
        checkpoint_frequency=0 if smoke else int(args.checkpoint_frequency),
        evaluation_frequency=0 if smoke else int(args.evaluation_frequency),
        evaluation_episodes=int(args.evaluation_episodes),
        tensorboard=bool(args.tensorboard),
        start_phase=args.start_phase,
        finetune_level_26=bool(args.finetune_level_26),
        wire_to_ground_probability=float(args.wire_to_ground_probability),
        smoke_test=smoke,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    started = time.perf_counter()
    model_path = train(config)
    if config.tensorboard:
        print(f"TensorBoard: tensorboard --logdir {config.output_dir / 'tensorboard'}")
    print(f"Model: {model_path}")
    print(f"Elapsed: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
