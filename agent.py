import flappy_bird_gymnasium
import gymnasium as gym
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools # for indefinite propagation
import yaml
import torch.nn  as nn
import random



if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Agent:
    def __init__(self,param_set):
        with open("parameters.yaml","r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]
        self.gamma = params["gamma"]
        self.env_id = params["env_id"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        

    def run(self, is_training=True, render = False):
  
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)# this becomes true for testing and false for training

        num_states = env.observation_space.shape[0] #input dim
        num_actions = env.action_space.n #output dimension (output_dim)

        policy_dqn = DQN(num_states, num_actions).to(device)

        state, _ = env.reset()

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_rewards = 0

            while not terminated:
                if is_training and random.random()<epsilon:
                    action = env.action_space.sample() #explore
                    action = torch.tensor(action, dtype=torch.long, device=device)

                else:
                    with torch.no_grad:
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() #exploit

                # Processing: terminated => done
                new_state, reward, terminated, _, _ = env.step(action)

                # creation of tensor 
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)


                if is_training:
                    memory.append((state, action,new_state, reward,terminated))

                state = new_state
                episode_rewards += reward

            print(f"for episode={episode+1}with total reward={episode_rewards} & epsilon={epsilon}")

            # epsilon decay
            epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)


            #env.close() -- for manually close