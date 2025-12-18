import numpy as np

from env.tnf_env import TNFEnv
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent

from agents.rl_agent import RLAgent
from stable_baselines3 import PPO

def run_episode(env, agents, render=False):
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        pid = env.agent_id
        action = agents[pid].select_action(env.game, pid)
        
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        if render:
            env.render()

    return total_reward

def evaluate(num_episodes: int = 100, render: bool = False, model_path: str = "models/stage_random/tnf_random_300000steps.zip"):
    env = TNFEnv()
    
    # Load trained model
    model = PPO.load(model_path, device='cpu')
    
    # change agents here
    agents = {
        0: RLAgent(model),
        1: RandomAgent(),
        2: RandomAgent(),
        3: GreedyAgent(),
    }
    
    rewards = []
    for _ in range(num_episodes):
        rewards.append(run_episode(env, agents, render))
        
    print(f"Mean reward over {num_episodes} episodes: {np.mean(rewards)}")
    print(f"Std: {np.std(rewards)}")
    

if __name__ == "__main__":
    evaluate(500) # run agent over 500 games