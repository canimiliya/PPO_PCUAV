import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from .assets import pretrained_model_path
from .environment import Scenario
from .simulation import metrics, run_baseline, run_policy
from .visualization import save_comparison, save_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pretrained hierarchical PPO-PID navigation demo."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=pretrained_model_path(),
        help="Stable-Baselines3 PPO model archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "outputs",
        help="Directory for the GIF, comparison figure, and metrics.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RRT* random seed.")
    parser.add_argument("--wire", type=int, choices=range(4), default=0)
    parser.add_argument(
        "--mode",
        choices=("ground_to_wire", "wire_to_ground"),
        default="ground_to_wire",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    scenario = Scenario(mode=args.mode, wire_index=args.wire)
    model = PPO.load(model_path, device="cpu")
    logs = [run_policy(model, scenario), run_baseline("A*", scenario)]

    rrt_error = None
    for offset in range(5):
        try:
            logs.append(run_baseline("RRT*", scenario, seed=args.seed + offset))
            rrt_error = None
            break
        except RuntimeError as error:
            rrt_error = error
    if rrt_error is not None:
        raise rrt_error

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = save_gif(logs[0], scenario, output_dir / "ppo_navigation.gif")
    figure_path = save_comparison(
        logs, scenario, output_dir / "trajectory_comparison.png"
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics(logs), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"GIF: {gif_path}")
    print(f"Comparison: {figure_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
