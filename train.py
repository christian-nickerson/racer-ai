from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from racer import RacingGym

TIMESTEPS = 100000
EPISODES = 1000
MODEL_NAME = "PPO"


if __name__ == "__main__":

    env = RacingGym()
    env.reset()

    model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="logs/")

    for ep in range(1, EPISODES):
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=MODEL_NAME,
        )
        model.save(f"artifacts/{MODEL_NAME}/{ep}")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    env.close()
