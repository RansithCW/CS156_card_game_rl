import numpy as np
from matplotlib import pyplot as plt

from env.tnf_env import TNFEnv
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent

from agents.rl_agent import RLAgent
from sb3_contrib import MaskablePPO

def run_episode(env, agents, render=False):
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        pid = env.agent_id
        
        obs = env._get_obs(pid)
        action = agents[pid].select_action(obs, info.get("action_mask"))  # Ensure opponents have played if needed
        
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        if render:
            env.render()

    return total_reward

def evaluate(
    num_episodes=100, 
    render=False, 
    model_path="models/stage_random/tnf_random_300000steps.zip", 
    plot_hist=False,
    game_type='random'
    ):
    env = TNFEnv()
    
    # Load trained model
    model = MaskablePPO.load(model_path, device='cpu')
    
    # change agents here
    agents = {
        0: RLAgent(model),
        1: RandomAgent(),
        2: RandomAgent(),
        3: RandomAgent(),
    }
    
    if game_type == 'greedy':
        agents[1] = GreedyAgent()
        agents[2] = GreedyAgent()
        agents[3] = GreedyAgent()
    
    elif game_type == 'mixed':
        agents[1] = GreedyAgent()
    
    rewards = []
    for _ in range(num_episodes):
        rewards.append(run_episode(env, agents, render))
        
    if plot_hist:
        # plot histogram of rewards
        plt.hist(rewards, bins=20)
        plt.title(f"Evaluation over {num_episodes} episodes")
        plt.xlabel("Total Reward")
        plt.ylabel("Frequency")
        plt.show()
        
    
    print(f"Mean reward over {num_episodes} episodes: {np.mean(rewards)}")
    print(f"Std: {np.std(rewards)}")
    
    return np.mean(rewards)
    

if __name__ == "__main__":
    evaluate(500) # run agent over 500 games