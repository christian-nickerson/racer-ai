from agents import AgentFactory, BaseAlgorithm, ScoreBook
from games import RacingGym
from uuid import uuid4
import shutil

TIMESTEPS = 10000
EPISODES = 500
MODEL_NAME = "PPO"


class RacingTrainer:

    """Train an RL agent to play a racing game"""

    def __init__(self, agent_name: str) -> None:
        """intialise trainer"""
        self.env = RacingGym()
        self.agent_name = agent_name
        self.agent = self._get_agent(self.agent_name, self.env)
        self.score_book = ScoreBook()

    def _get_agent(self, agent_name: str, env: RacingGym) -> BaseAlgorithm:
        """Get agent from Agent Factory

        :param agent_name: Name rof agent to return from the factory
        :param env: game environment to instantite the agent with
        :return: Agent from Agent Factory
        """
        factory = AgentFactory()
        agent = factory.get_agent(agent_name)
        return agent("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="logs/")

    def _compress_artifacts(self, batch_id: str) -> None:
        """compress directory of agent artifacts

        :param batch_id: batch_id to compress artifacts for
        """
        path = f"artifacts/{self.agent_name}/{batch_id}"
        shutil.make_archive(path, "tar", path)
        shutil.rmtree(path)

    def train(self, timesteps: int, epidoses: int) -> None:
        """Train agent on racing game.

        :param timesteps: number of game timesteps to learn from
        :param epidoses: number of rounds to play the game
        """
        self.env.reset()
        batch_id = uuid4()

        for ep in range(epidoses):
            total_timesteps = (ep + 1) * timesteps
            self.agent.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=False,
                tb_log_name=self.agent_name,
            )
            self.score_book.record_result(self.agent, self.agent_name, batch_id, ep, total_timesteps)
            self.agent.save(f"artifacts/{self.agent_name}/{batch_id}/{total_timesteps}")

        self._compress_artifacts(batch_id)
        self.score_book.write_parquet()
        self.env.close()


if __name__ == "__main__":

    racing_trainer = RacingTrainer(MODEL_NAME)
    racing_trainer.train(TIMESTEPS, EPISODES)
    print(racing_trainer.score_book.ledger)
