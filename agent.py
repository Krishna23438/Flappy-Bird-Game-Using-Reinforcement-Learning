import flappy_bird_gymnasium
import gymnasium as gym
import torch
from dqn import DQN

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

    while True:
      # Next action:
      # (feed the observation to your agent here)
      action = env.action_space.sample()

      # Processing: terminated => done
      next_state, reward, terminated, _, _ = env.step(action)
    
      # Checking if the player is still alive
      if terminated:
          break

    env.close()