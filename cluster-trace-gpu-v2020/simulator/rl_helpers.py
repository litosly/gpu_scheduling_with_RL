import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

# Define the network structure
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the experience tuples
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

# Define the Q-Learning agent using the DQN (Deep Q-Network) algorithm
class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size=10000, batch_size=64, seed=seed)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        # Returns actions for given state as per current policy.
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming a state and action space size (dummy values, replace with actual sizes)
state_size = 10  # The size of the job representation
action_size = 4  # Assuming four possible nodes to allocate

agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)


# def get_state(cluster):
#     num_available_gpus = cluster.get_num_available_gpus()
#     job_queue_length = len(cluster.job_queue)
#     average_wait_time = cluster.get_average_wait_time()
#     group_gpu_dur = cluster.get_group_gpu_dur() # You need to define how to calculate this
#     return (num_available_gpus, job_queue_length, average_wait_time, group_gpu_dur)
def select_action(job_queue):
    # Sort jobs by group_gpu_dur and select the job with the shortest time
    sorted_jobs = sorted(job_queue, key=lambda x: x.group_gpu_dur)
    return sorted_jobs[0] if sorted_jobs else None
def get_reward(job):
    # Negative of the job completion time
    return -job.completion_time
def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    max_future_q = max(q_table[next_state, :])
    current_q = q_table[state, action]
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    q_table[state, action] = new_q


# Assuming that the Cluster class has methods to get the number of available GPUs, job queue, and average wait time
# The following is a pseudocode implementation of the state representation function
# Note: You will need to replace the method names with the actual method names from your Cluster class

def get_state(cluster):
    # Assuming cluster has a method to get the number of available GPUs
    # num_available_gpus = cluster.get_num_available_gpus() 
    
    # Assuming cluster has a property or method to get the job queue
    job_queue_length = len(cluster.job_queue)
    
    # Assuming cluster has a method to calculate the average wait time of jobs in the queue
    average_wait_time = cluster.get_average_wait_time()
    
    # Assuming cluster has a method to get the current 'group_gpu_dur' estimate
    # This might be a calculated property based on the current jobs in the queue or running
    group_gpu_dur = cluster.get_group_gpu_dur()
    
    # The state is a tuple of these features
    return (num_available_gpus, job_queue_length, average_wait_time, group_gpu_dur)

# This is a placeholder function to represent the actual Cluster class methods
# You will need to implement these methods in your Cluster class or modify this function
# to correctly interact with your Cluster class implementation

# For now, let's define mock methods to test our state representation function

class MockCluster:
    def __init__(self):
        self.job_queue = []  # Mock job queue
        # Mock method to get the number of available GPUs
        self.get_num_available_gpus = lambda: 10
        # Mock method to get the average wait time
        self.get_average_wait_time = lambda: 5.0
        # Mock method to get the group_gpu_dur
        self.get_group_gpu_dur = lambda: 120

# Create a mock cluster object
mock_cluster = MockCluster()

# Get the state representation
state = get_state(mock_cluster)
state



def get_reward(job):
    # The reward is the negative of the Job Completion Time (JCT)
    return -job.completion_time

# Placeholder class to represent job objects with a 'completion_time' attribute
class MockJobWithCompletionTime:
    def __init__(self, completion_time):
        self.completion_time = completion_time

# Create a mock job with a specific completion time to test the reward function
mock_job = MockJobWithCompletionTime(completion_time=120)

# Calculate the reward for the mock job
reward = get_reward(mock_job)
reward

# Given that the job queue is a list of job objects and each job has a 'group_gpu_dur' attribute
# We can implement the action selection function as follows:

def select_action(job_queue):
    # If the job queue is empty, there is no action to take
    if not job_queue:
        return None
    
    # Sort jobs by 'group_gpu_dur' and select the job with the shortest estimated time
    # Assuming that smaller 'group_gpu_dur' values are better (shorter estimated time)
    selected_job = min(job_queue, key=lambda job: job.group_gpu_dur)
    return selected_job

# This is a placeholder class to represent the actual job objects
# You will need to ensure that your job objects have a 'group_gpu_dur' attribute

class MockJob:
    def __init__(self, group_gpu_dur):
        self.group_gpu_dur = group_gpu_dur

# Create a mock job queue with mock jobs
mock_job_queue = [MockJob(group_gpu_dur=100), MockJob(group_gpu_dur=200), MockJob(group_gpu_dur=50)]

# Select an action (job) from the mock job queue
selected_job = select_action(mock_job_queue)

# The selected job should be the one with the shortest 'group_gpu_dur'
selected_job.group_gpu_dur



def update_q_table(q_table, state, action, reward, next_state, alpha, gamma, all_actions):
    # Find the max Q-value of the next state
    max_future_q = max(q_table[next_state][a] for a in all_actions)

    # Current Q-value for the state-action pair
    current_q = q_table[state][action]

    # Calculate the new Q-value
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)

    # Update the Q-table with the new Q-value
    q_table[state][action] = new_q
