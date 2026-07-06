import flappy_bird_gymnasium
import gymnasium as gym
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools # for indefinite propagation

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def run(self, is_training=True, render = False):
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None)# this becomes true for testing and false for training

    num_states = env.observation_space.shape[0] #input dim
    num_actions = env.action_space.n #output dimension (output_dim)

    policy_dqn = DQN(num_states, num_actions).to(device)

    state, _ = env.reset()

    if is_training:
        memory = ReplayMemory(10000)

    for episode in itertools.count():
        while True:
            state, _ = env.reset()
            terminated = False
            episode_rewards = 0

            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing: terminated => done
            new_state, reward, terminated, _, _ = env.step(action)

            if is_training:
                memory.append((state, action,new_state, reward,terminated))

            state = new_state
            episode_rewards += reward

        print(f"for episode={episode+1}with total reward={episode_rewards}")
    
            
            
        #env.close() -- for manually close