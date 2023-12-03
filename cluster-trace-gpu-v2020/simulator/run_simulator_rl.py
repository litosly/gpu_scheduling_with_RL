# usage: python3 run_simulator.py & date

from simulator import Simulator
from utils import print_fn, ALLOC_POLICY_DICT, PREEMPT_POLICY_DICT
import os
import time
import logging
import argparse
from pathlib import Path

DATE = "%02d%02d" % (time.localtime().tm_mon, time.localtime().tm_mday)

# INPUT TRACE FILE
CSV_FILE_PATH = Path(__file__).parent / 'traces/pai/'
DESCRIBE_FILE = None
CSV_FILE = 'pai_job_duration_estimate_100K.csv'

parser = argparse.ArgumentParser(description='Simulator.')
parser.add_argument("-r", "--arrival_rate", help="Arrival Rate", type=int, default=1000)
parser.add_argument("-n", "--num_jobs", help="Num of Jobs", type=int, default=20000)
parser.add_argument("-g", "--num_gpus", help="Num of GPUs", type=int, default=6500)
parser.add_argument("-p", "--repeat", help='Repeat', type=int, default=1)
parser.add_argument("-k", "--pack", dest='packing_policy', action='store_true')
parser.add_argument("-b", "--balance", dest='packing_policy', action='store_false')
parser.set_defaults(packing_policy=False)
args = parser.parse_args()
NUM_JOBS = args.num_jobs
ARRIVAL_RATE = args.arrival_rate
NUM_GPUS = args.num_gpus
REPEAT = args.repeat
SORT_NODE_POLICY = 0 if args.packing_policy is True else 3  # 0: packing, 3: max-min balancing.

MAX_TIME = int(1e9)

VERBOSE = 0
# VERBOSE = 1
# LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.WARNING

NUM_NODES = 1
NUM_CPUS = round(23.22 * NUM_GPUS)  # 23.22 * num_gpus 156576/6742
# HETERO = True  # heterogeneous cluster
HETERO = False
PATTERN = 0  # Cluster capacity varying pattern

# GPU_TYPE_MATCHING = 1 # GPU type perfect match
# GPU_TYPE_MATCHING = 2 # Only V100 cannot compromise
GPU_TYPE_MATCHING = 0

EXPORT_JOB_STATS = False
# EXPORT_JOB_STATS = True
EXPORT_CLUSTER_UTIL = False
# EXPORT_CLUSTER_UTIL = True

# RANDOM_SEED = random.randint(0, 100)
RANDOM_SEED = 42
NUM_SPARE_NODE = 0
# SORT_BY_JCT = False
SORT_BY_JCT = True

# Logging in directory
LOG_DIR = Path(__file__).parent / 'logs'

comments = '%dg_%dn_h%d_%dp_%dsn_%dgt-%dar-%dj-%dx-%dr' % (NUM_GPUS, NUM_NODES, HETERO, PATTERN, SORT_NODE_POLICY, GPU_TYPE_MATCHING, ARRIVAL_RATE, NUM_JOBS, REPEAT, RANDOM_SEED)

log_time = int(time.time() % 100000)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file = LOG_DIR / ("%s-%s-%s-%s.log" % (DATE, CSV_FILE, log_time, comments))
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", filename=log_file, filemode='a')
describe_file = CSV_FILE_PATH / DESCRIBE_FILE if DESCRIBE_FILE is not None else None

results_dict = {}
num_jobs_dict = {}
avg_jct_dict = {}
makespan_dict = {}
wait_time_dict = {}
runtime_dict = {}

print("log_file: %s" % log_file)
print_str = "==========\n%d_Jobs_repeated_%d_times\nalloc,preempt,avg_jct,wait_time,makespan,jobs_done,runtime" % (NUM_JOBS, REPEAT)
print(print_str)
print_fn(print_str, level=2)


# Assuming 'DQNAgent' and 'Simulator' are defined as per previous examples

# Initialize your simulator as before
simulator = Simulator(
    csv_file=CSV_FILE_PATH / CSV_FILE,
    alloc_policy=alloc_policy,
    preempt_policy=preempt_policy,
    sort_node_policy=SORT_NODE_POLICY,
    num_nodes=NUM_NODES,
    random_seed=RANDOM_SEED,
    max_time=MAX_TIME,
    num_spare_node=NUM_SPARE_NODE,
    pattern=PATTERN,
    hetero=HETERO,
    num_gpus=NUM_GPUS,
    num_cpus=NUM_CPUS,
    describe_file=describe_file,
    log_file=log_file,
    export_job_stats=EXPORT_JOB_STATS,
    export_cluster_util=EXPORT_CLUSTER_UTIL,
    arrival_rate=ARRIVAL_RATE,
    num_jobs_limit=NUM_JOBS,
    gpu_type_matching=GPU_TYPE_MATCHING,
    verbose=VERBOSE
    )

# Assuming you have a function to convert the simulator state to a neural network-friendly format
def extract_state(simulator):
    # This function needs to be defined by you. It should convert the simulator's current state
    # into the input format expected by your neural network (state_size-dimensional vector).
    # For example, you might extract information about the current jobs and their statuses.
    # return state_as_vector

    # Assuming the simulator provides a method to get the current status of jobs and resources
    # If the simulator doesn't provide it directly, you may need to implement such methods.
    
    # Normalize job attributes
    jobs_data = []
    if simulator.job_list:  # Ensure there are jobs to get data from
        # Get max values for normalization
        max_num_gpus = max(job.num_gpu for job in simulator.job_list)
        max_num_cpus = max(job.num_cpu for job in simulator.job_list)
        max_duration = max(job.duration for job in simulator.job_list)
        max_wait_time = max(job.wait_time for job in simulator.job_list)
        max_submit_time = max(job.submit_time for job in simulator.job_list)  # Assume submit_time is in some normalized format
        
        # Extract and normalize job data
        for job in simulator.job_list:
            jobs_data.extend([
                job.num_gpu / max_num_gpus if max_num_gpus else 0,
                job.num_cpu / max_num_cpus if max_num_cpus else 0,
                job.duration / max_duration if max_duration else 0,
                job.wait_time / max_wait_time if max_wait_time else 0,
                job.submit_time / max_submit_time if max_submit_time else 0,
                # Add more attributes if necessary
            ])

    # Flatten the jobs data and pad with zeros to match the input size of the network
    max_jobs_count = 10  # set the maximum number of jobs you want to consider
    job_features_length = max_jobs_count * 5  # assuming we're using 5 features per job
    if len(jobs_data) < job_features_length:
        # Pad with zeros if the current job list is shorter than the max length
        jobs_data += [0] * (job_features_length - len(jobs_data))
    elif len(jobs_data) > job_features_length:
        # Truncate the list if it exceeds the max length
        jobs_data = jobs_data[:job_features_length]

    # Convert to numpy array
    state = np.array(jobs_data, dtype=np.float32)
    return state


# Define the action_size if not done already - it must match your neural network output
action_size = simulator.num_nodes  # or some other number based on your simulator's action space

# Initialize your RL agent
state_size = ...  # define the size of the state vector that your network expects
agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)

# Define a function to choose an action using the RL agent
def choose_action(state):
    # Convert state to the appropriate tensor format
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    # Use the agent to choose an action
    action = agent.act(state_tensor, eps=0.1)  # eps can be decayed over time
    return action

# Training loop
for i_episode in range(num_episodes):
    # Reset the simulator
    # (You may need to implement or modify a reset method within the Simulator class)
    simulator.reset()

    # Initialize the state (convert the current simulator state to a vector)
    state = extract_state(simulator)

    while True:  # Run the episode until it's done
        # Select an action
        action = choose_action(state)

        # Apply action to the simulator and get the next state and reward
        # (You need to implement this interaction in the Simulator class)
        reward, next_state, done = simulator.step(action)  # Modify the simulator to return these values

        # Save the experience in the replay memory
        agent.step(state, action, reward, next_state, done)

        # Update state
        state = next_state

        # End the episode if done
        if done:
            break

    # Print episode stats
    print(f"Episode {i_episode}/{num_episodes} | Reward: {episode_reward}")

    # Perform any necessary updates to the agent here, like epsilon decay
    # ...

# Save the trained Q-network
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
