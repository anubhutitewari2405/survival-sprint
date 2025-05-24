#train dqn agent
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from red_green_env import RedGreenEnv

def train():
    env = RedGreenEnv()
    check_env(env)

    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("red_green_dqn_model")

def test():
    env = RedGreenEnv()
    model = DQN.load("red_green_dqn_model")

    obs, info = env.reset()
    for _ in range(20):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            break

if __name__ == "__main__":
    train()
    print("\nTesting trained agent:\n")
    test()
