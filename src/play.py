from agents import RLRegistry
from stable_baselines3 import PPO
from games import RacingGym

MODEL_NAME = "PPO"
LOAD_EPISODE = 498

artifacts_path = f"artifacts/{MODEL_NAME}"
model_path = f"~./{artifacts_path}/{LOAD_EPISODE * 10000}"

print(model_path)

env = RacingGym()
env.reset()

model_registry = RLRegistry()
model_instance = model_registry.get_model(MODEL_NAME, env)
model = PPO.load("/home/christian_nickerson/python/racer_ai/artifacts/PPO/12200000.zip", env=env, device="cpu")

if __name__ == "__main__":

    observation = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(observation)
        obs, reward, done, info = env.step(action)

    env.close()
