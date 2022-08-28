from stable_baselines3 import PPO, A2C, TD3, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
from pathlib import Path
from typing import Any
from gym import Env


class RLRegistry:

    """Reinforcement Model registry"""

    def __init__(self) -> None:
        """Reinforcement Model registry"""
        self.registry = {}
        self._compile_registry()

    def _logging_path(self) -> None:
        """define & create logging path"""
        self.log_path = "logs/"
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, model_name: str) -> None:
        """define & create artifact path"""
        self.artifact_path = f"artifacts/{model_name}"
        Path(self.artifact_path).mkdir(parents=True, exist_ok=True)

    def _register_model(self, model: Any) -> None:
        """register a model within the registry"""
        self.registry[model.__name__] = model

    def _compile_registry(self) -> None:
        """compile models into registry"""
        self._register_model(PPO)
        self._register_model(A2C)
        self._register_model(TD3)
        self._register_model(DDPG)

    def get_model(self, model_name: str, env: Env) -> BaseAlgorithm:
        """get model from registry"""
        self._artifact_path(model_name)
        self._logging_path()
        return self.registry[model_name]("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=self.log_path)


if __name__ == "__main__":
    pass
