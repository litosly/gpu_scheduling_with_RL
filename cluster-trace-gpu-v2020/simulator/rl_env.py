"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

# import gym
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
import statistics


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

# parser = argparse.ArgumentParser(description='Simulator.')
# parser.add_argument("-r", "--arrival_rate", help="Arrival Rate", type=int, default=1000)
# parser.add_argument("-n", "--num_jobs", help="Num of Jobs", type=int, default=20000)
# parser.add_argument("-g", "--num_gpus", help="Num of GPUs", type=int, default=6500)
# parser.add_argument("-p", "--repeat", help='Repeat', type=int, default=1)
# parser.add_argument("-k", "--pack", dest='packing_policy', action='store_true')
# parser.add_argument("-b", "--balance", dest='packing_policy', action='store_false')
# parser.set_defaults(packing_policy=False)
# args = parser.parse_args()

# NUM_JOBS = args.num_jobs
# ARRIVAL_RATE = args.arrival_rate
# NUM_GPUS = args.num_gpus
# REPEAT = args.repeat
# SORT_NODE_POLICY = 0 if args.packing_policy is True else 3  # 0: packing, 3: max-min balancing.

packing_policy = False
NUM_JOBS = 20000
ARRIVAL_RATE = 1000
NUM_GPUS = 6500
REPEAT = 1
SORT_NODE_POLICY = 0 if packing_policy is True else 3  # 0: packing, 3: max-min balancing.


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



class GPUJobEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the strategy
    for choosing the next available job from the job list

    | Num | Action                 
    |-----|------------------------
    | 0   | pick next shortest job                  
    | 1   | pick next job with smallest submit time 

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | average wait time     |  0                  | Inf               |
    | 1   | length of job list    |  0                  | Inf               |
    | 2   | min wait time         |  0                  | Inf               |
    | 3   | max wait time         |  0                  | Inf               |

    ### Rewards
    negative of job completion time whenever there is a job completed, 0 for no update

    ### Starting State

    [0, 0, 0, 0]

    ### Episode End

    The episode ends if :

    1. Termination: len(self.job_full_list) <= 0

    ### Arguments

    ```
    gym.make('GPUJob-v1')
    ```

    """

    def __init__(self, render_mode: Optional[str] = None):
        self.simulator = Simulator(
            csv_file=CSV_FILE_PATH / CSV_FILE,
            alloc_policy=14, # RL
            preempt_policy=2,
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
            verbose=VERBOSE)
        self.simulator.init_go()
        self.result = []

        min_avg_wait_time = 0
        min_avg_duration = 0
        min_job_list_length = 0
        min_max_wait_time = 0
        min_min_duration = 0
        max_avg_wait_time = float('inf')
        max_avg_duration = float('inf')
        max_job_list_length = float('inf')
        max_max_wait_time = float('inf')
        max_min_duration = float('inf')

        min_values = np.array([min_avg_wait_time, min_avg_duration, min_job_list_length, min_max_wait_time, min_min_duration])
        max_values = np.array([max_avg_wait_time, max_avg_duration, max_job_list_length, max_max_wait_time, max_min_duration])

        # Create the action space and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=min_values, high=max_values, dtype=np.float32)

        self.state = None

    def step(self, action, delta = 1):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        terminated = 0
        # Read current state
        avg_wait_time, avg_duration, job_list_length, max_wait_time, min_duration = self.state
        
        ## Move forward
        # update state
        # terminate state
        # reward 
        
        # Actual Run (One tic)
        # self.tic(self.delta)
        if self.simulator.cur_time < self.simulator.max_time:
            self.simulator.cluster.tic_svc(self.simulator.cur_time) #normally won't update since we have pattern = 0

            # Preempt job
            self.simulator.scheduler.preempt_job(self.simulator.cluster)

            # Allocate job based on action
            # action can be 0 (SJF) or 1(FIFO)
            self.simulator.scheduler.alloc_job(self.simulator.cluster, action=action)

            # Jobs tic and global cur_time += delta
            tic_return_value, reward = self.simulator.cluster.tic_job(delta, return_reward=True)

            # After one tic
            if tic_return_value >= 0:
                self.simulator.cur_time = tic_return_value
            else:
                self.simulator.cur_time = self.simulator.cur_time + delta
                self.simulator.exit_flag = 1
                terminated = 1
        else:
            print_fn("TIMEOUT {} with jobs {}".format(self.cur_time, self.cluster.job_list))
            self.exit_flag = 1
            raise TimeoutError("TIMEOUT {} with jobs {}".format(self.cur_time, self.cluster.job_list))
        
        
        if self.simulator.exit_flag:
            terminated = 1
            num_jobs_done, jct_summary, wait_time_summary = self.simulator.exp_summary(0)
            self.result.append((num_jobs_done, jct_summary / num_jobs_done, wait_time_summary / num_jobs_done, self.simulator.cur_time))
            # avg_wait_time = wait_time_summary / num_jobs_done
            # avg_duration = jct_summary / num_jobs_done
        
        # Update New State
        job_list = self.simulator.cluster.job_list
        if self.simulator.cur_time % 10000 == 0:
            print("current time: ", self.simulator.cur_time)
            print("current job list in cluster: ", len(job_list))
            print("current running job: ", len(self.simulator.cluster.job_runn_list))
            print("length of the full list: ", len(self.simulator.cluster.job_full_list))

        job_list_length = len(job_list)

        wait_time_list = [self.simulator.cur_time - job["submit_time"] for job in job_list]
        if wait_time_list:
            avg_wait_time = statistics.mean(wait_time_list)
            max_wait_time = max(wait_time_list)

        duration_list = [job["group_gpu_dur"] for job in job_list]
        if duration_list:
            avg_duration = statistics.mean(duration_list)
            min_duration = min(duration_list)

        self.state = (avg_wait_time, avg_duration, job_list_length, max_wait_time, min_duration)

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        self.state = np.zeros(5)
        return np.array(self.state, dtype=np.float32), {}

    # def close(self):
    #     if self.screen is not None:
    #         import pygame

    #         pygame.display.quit()
    #         pygame.quit()
    #         self.isopen = False