# usage: python3 run_simulator.py & date

from simulator import Simulator
from utils import print_fn, ALLOC_POLICY_DICT, PREEMPT_POLICY_DICT
import os
import time
import logging
import argparse
from pathlib import Path
from rl_env import GPUJobEnv
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
import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

env = GPUJobEnv()

# model = A2C("MlpPolicy", env, verbose=1)
# print("start learning")
# model.learn(total_timesteps=100000)
# print("learning done")
# model.save("reward_2") # no throughput reward
# model.save("reward_3") # avg done penalty  + throughput
# model.save("reward_4") # avg waiting penalty for cluster job list + throughput
# model.save("reward_5") # avg done penalty + avg waiting penalty for cluster job list + throughput
# print("model saved")

## Load and evaluate model
model = A2C.load("reward_5", env=env)

## not important evaluation
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
# print("mean_reward, std_reward: ", mean_reward, std_reward)

vec_env = model.get_env()
obs = vec_env.reset()
action_list = []
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    action_list.append(action)
    # print("action: ", action)
    obs, reward, done, info = vec_env.step(action)
    # if done:
    #   obs = vec_env.reset()

print("obs: ", obs)
print("reward: ", reward)
print("done: ", done)
print("info: ", info)
print("action list sum: ", sum(action_list))