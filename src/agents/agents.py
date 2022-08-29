from stable_baselines3 import PPO, A2C, TD3, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
from pathlib import Path


class AgentFactory:

    """Reinforcement agent factory"""

    def __init__(self) -> None:
        """Reinforcement agent registry"""
        self.registry = {}
        self._compile_registry()

    def _logging_path(self) -> None:
        """define & create logging path"""
        self.log_path = "logs/"
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, model_name: str) -> None:
        """define & create artifact path

        :param model_name: name of model to write path to
        """
        self.artifact_path = f"artifacts/{model_name}"
        Path(self.artifact_path).mkdir(parents=True, exist_ok=True)

    def _register_agent(self, agent: BaseAlgorithm, agent_name: str) -> None:
        """register a agent in the factory

        :param agent: Agent to register
        """
        self.registry[agent_name] = agent

    def _compile_registry(self) -> None:
        """compile agent into registry"""
        self._register_agent(PPO, "PPO")
        self._register_agent(A2C, "A2C")
        self._register_agent(TD3, "TD3")
        self._register_agent(DDPG, "DDPG")

    def get_agent(self, agent_name: str) -> BaseAlgorithm:
        """get agent from factory

        :param agent_name: name of agent to return from factory
        :return: Agent object from factory
        """
        self._artifact_path(agent_name)
        self._logging_path()
        return self.registry[agent_name]
