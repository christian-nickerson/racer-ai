from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from pathlib import Path
import pandas as pd


class ScoreBook:

    """Agent scoring book"""

    ledger: pd.DataFrame

    def __init__(self) -> None:
        self.ledger = pd.DataFrame(
            columns=[
                "agent_name",
                "mean_reward",
                "std_reward",
                "batch_id",
                "episode",
                "total_steps",
            ]
        )

    def record_result(
        self,
        agent: BaseAlgorithm,
        agent_name: str,
        batch_id: int,
        episode: int,
        total_steps: int,
    ) -> None:
        """record agent's training score

        :param agent: agent to be evaluated
        :param agent_name: name of agent being evauluated
        :param batch_id: ID of the training batch run
        :param episode: training episode at time of evaluation
        :param total_steps: total training steps agent has seen
        """
        mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10)
        result = pd.DataFrame(
            {
                "agent_name": [agent_name],
                "mean_reward": [mean_reward],
                "std_reward": [std_reward],
                "batch_id": [str(batch_id)],
                "episode": [episode],
                "total_steps": [total_steps],
            }
        )
        self.ledger = pd.concat([self.ledger, result])

    def write_parquet(self) -> None:
        """write score book to parquet file"""
        path = "scores/"
        filename = "scorebook-{i}.parquet"
        Path(path).mkdir(parents=True, exist_ok=True)
        existing_ledger = pd.read_parquet(path)
        self.ledger = pd.concat([existing_ledger, self.ledger]).drop_duplicates(keep="last")
        self.ledger.to_parquet(path, partition_cols=["agent_name"], basename_tempplate=filename)
