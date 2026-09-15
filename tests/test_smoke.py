from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ppo_pcuav.assets import pretrained_model_path
from ppo_pcuav.environment import PayloadUAVEnv, Scenario
from ppo_pcuav.training import required_success_rate
from ppo_pcuav.training_environment import CurriculumPayloadUAVEnv


def test_environment_contract():
    environment = PayloadUAVEnv()
    check_env(environment, warn=True)
    observation, info = environment.reset(
        seed=0, options={"scenario": Scenario("ground_to_wire", 0)}
    )
    assert observation.shape == (18,)
    assert info["task_mode"] == "ground_to_wire"

    next_observation, _, terminated, truncated, step_info = environment.step([0, 0])
    assert next_observation.shape == (18,)
    assert not (terminated and truncated)
    assert step_info["step_trajectory"].shape[1] == 8


def test_pretrained_model_contract():
    model_path = pretrained_model_path()
    assert model_path.is_file()
    model = PPO.load(model_path, device="cpu")
    environment = PayloadUAVEnv()
    observation, _ = environment.reset(options={"scenario": Scenario()})
    action, _ = model.predict(observation, deterministic=True)
    assert model.observation_space.shape == environment.observation_space.shape
    assert environment.action_space.contains(action)


def test_curriculum_environment_contract():
    environment = CurriculumPayloadUAVEnv()
    check_env(environment, warn=True)
    environment.set_level(20)
    observation, info = environment.reset(
        seed=0,
        options={
            "task_mode": "wire_to_ground",
            "target_wire_index": 2,
            "ground_x_min": 45.0,
            "ground_x_max": 45.0,
        },
    )
    assert observation.shape == (18,)
    assert info["level"] == 20
    assert info["task_mode"] == "wire_to_ground"
    assert info["target_wire_index"] == 2
    assert required_success_rate(20) == 0.75
    assert required_success_rate(26) == 0.95
