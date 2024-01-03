from collections import OrderedDict
from node import Node
from utils import print_fn, _repr_job_preempt, _repr_job_done, large_job_pruning
from job_history import JobHistory

import numpy as np

def normalize_wait_time(wait_time, min_wait_time, max_wait_time):
    # Avoid division by zero in case max and min are the same
    if max_wait_time == min_wait_time:
        return 0
    else:
        return (wait_time - min_wait_time) / (max_wait_time - min_wait_time)
    
class Cluster:
    def __init__(self, node_list=None, num_nodes=None, num_gpus=20,
                 num_cpus=20, pattern=1, period=124, job_list=None,
                 random_seed=0, num_spare_node=None,
                 export_cluster_util=False):
        if node_list is not None:
            node_list = node_list
        elif num_nodes is not None:
            node_list = [Node(id=i) for i in range(num_nodes)]
        else:
            node_list = [Node(id=0, num_gpus=num_gpus, num_cpus=num_cpus)]

        temp_node_dict = dict()
        self.num_gpus, self.num_cpus = 0, 0
        for node in node_list:
            self.num_gpus += node.num_gpus
            self.num_cpus += node.num_cpus
            temp_node_dict[node.id] = node
        self.node_dict = OrderedDict(sorted(temp_node_dict.items(),
                                            key=lambda t: t[1].id))

        self.cur_time = 0
        self.svc = {'num_gpu': 0, 'num_cpu': 0} # high-priority service
        self.svc_former_ratio = 0

        # self.job_full_list = job_list  # all jobs received from all times
        self.job_full_list = large_job_pruning(job_list, self.num_gpus, self.num_cpus)
        self.job_full_list.sort(key=lambda j: -j['submit_time'])
        self.job_list = []
        self.retrieve_job_from_full_list()  # feed self.user_job_queue into self.job_list

        self.job_history = JobHistory()

        # Capacity changing pattern & period
        self.pattern = pattern
        self.period = period

        # Spare specific node
        self.num_spare_node = num_spare_node
        self.spare_node_id = []
        if num_spare_node is not None:
            for i in range(num_spare_node):
                spare_node_index = random_seed % len(node_list)
                spare_node_id = node_list[spare_node_index].id
                while spare_node_id in self.spare_node_id:
                    random_seed += 29741  # a random prime number
                    spare_node_index = random_seed % len(node_list)
                    spare_node_id = node_list[spare_node_index].id
                self.spare_node_id.append(spare_node_id) # indicate which node to spare
                random_seed += 29741  # a random prime number

        self.export_cluster_util = export_cluster_util
        self.cluster_time = []
        self.cluster_cpu = []
        self.cluster_gpu = []
        self.idle_cluster_counter = 0

    def retrieve_job_from_full_list(self):
        while len(self.job_full_list) > 0:
            job = self.job_full_list[-1]
            if job['submit_time'] <= self.cur_time:
                job = self.job_full_list.pop()
                self.job_list.append(job)
            else:
                return 0

    def sorted_node_list(self):
        node_list = list(self.node_dict.values())
        node_list.sort(key=lambda n: n.id)
        return node_list

    def tic_job(self, delta=1, return_reward=False, obs = None):
        # Unlike tic_svc(), it receives simulator's cur_time as its own cur_time
        # Here it returns a "cur_time" value to the simulator
        # If succeed: return cur_time >= 0
        # Else: return cur_time < 0 ==> exit_flag = 1

        reward = 0 
        # get previous num of done jobs
        prev_num_jobs_done = self.job_history.num_jobs_done

        self.cur_time += delta
        if self.export_cluster_util and self.cur_time % 10000 == 0:
            self.record_cluster_util()
        self.retrieve_job_from_full_list()  # update self.job_list
        job_runn_list = self.job_runn_list
        if len(job_runn_list) > 0:
            for job in job_runn_list:
                job['on_time'] += delta # unit time pass, 1 sec by default
                job['progress'] = job['on_time'] * job['num_gpu'] # num_gpu sec progress by default for every unit time pass
                
                # Job done logic
                if job['on_time'] >= job['duration']:
                    over_tic_time = job['on_time'] - job['duration']  # only if delta > 1
                    job['on_time'] -= over_tic_time
                    job['progress'] -= over_tic_time * job['num_gpu']
                    job['done'] = 1

                    host_node_id = job['node']
                    host_node = self.node_dict.get(host_node_id)
                    suc = host_node.release_job(job=job)
                    assert suc

                    job['jct'] = self.cur_time - over_tic_time - job['submit_time']  # deduct submit_time

                    self.job_history.add_done_job(job)
                    # print("self.cur_time: ", self.cur_time)
                    # print("over_tic_time: ", over_tic_time)
                    # print("job['submit_time']: ", job['submit_time'])
                    # print("self.job_history.jct_summary: ", self.job_history.jct_summary)
                    ################# Reward Engineering ###################
                    # reward -= job['jct'] # method 1
                    # reward reduction in queue length method 2
                    # reward += 1
                    # Compute average wait time
                    if return_reward:
                        if self.job_history.num_jobs_done:
                            avg_wait_time = self.job_history.wait_time_summary / self.job_history.num_jobs_done
                        else:
                            avg_wait_time = 0

                        wait_time = job['jct'] - job['duration']
                        extra_wait_time = wait_time - avg_wait_time
                        penalty = np.sqrt(extra_wait_time) if extra_wait_time > 0 else 0
                        reward = reward - penalty

                    print_fn("%sDONE: %s || %s" % (self.log_prefix, _repr_job_done(job), job))
            
            # Reward higher throughput, i.e. more jobs in a given time
            if return_reward:
                diff_num_jobs_done = self.job_history.num_jobs_done - prev_num_jobs_done
                wait_time_list = [self.cur_time - job["submit_time"] for job in self.job_list]
                if diff_num_jobs_done: # normalize done jobs waiting penalty
                    reward = reward / diff_num_jobs_done

                mean_wait_time = np.mean(wait_time_list)
                avg_waiting_time_penalty_for_cluster_job = np.sqrt(mean_wait_time) if mean_wait_time > 0 else 0
                reward -= avg_waiting_time_penalty_for_cluster_job # average waiting time penalty for cluster jobs
                
                reward += diff_num_jobs_done if diff_num_jobs_done > 0 else 0 # throughput reward
                return self.cur_time, reward
            
            # Update obs, i.e. rl state space if obs not None:
            if type(obs) != type(None):
                job_list = self.job_list

                num_jobs_in_cluster = len(job_list)
                wait_time_list = [self.cur_time - job["submit_time"] for job in self.job_list]
                if wait_time_list:
                    avg_wait_time_cluster = np.mean(wait_time_list)
                else:
                    avg_wait_time_cluster = 0

                duration_list = [job["group_gpu_dur"] for job in job_list]
                if duration_list:
                    avg_duration_estimate = np.mean(duration_list)
                else:
                    avg_duration_estimate = 0
                # Done jobs related
                job_history = self.job_history
                num_jobs_done = job_history.num_jobs_done
                if num_jobs_done:
                    avg_jct = job_history.jct_summary/num_jobs_done
                    avg_wait_time_done = job_history.wait_time_summary/num_jobs_done
                    avg_wasted_time = job_history.wasted_summary/num_jobs_done
                else:
                    avg_jct = 0
                    avg_wait_time_done = 0
                    avg_wasted_time = 0
                obs = (num_jobs_in_cluster, avg_wait_time_cluster, avg_duration_estimate, num_jobs_done, avg_jct, avg_wait_time_done, avg_wasted_time)
                return self.cur_time, obs
            return self.cur_time  # exit_flag = 0, still going

        # len(job_runn_list) <= 0,
        elif len(self.job_list) > 0:  # empty cluster with job pending
            self.idle_cluster_counter += 1
            print_fn("%sIDLE cluster until jobs: %s" % (self.log_prefix, [_repr_job_preempt(e) for e in self.job_list]))

            if self.idle_cluster_counter % 10000 == 0:
                print_fn('{} idle cluster: {}'.format(self.idle_cluster_counter, [_repr_job_preempt(e) for e in self.job_list]), level=2)
            if return_reward:
                return self.cur_time, reward
            if type(obs) != type(None):
                return self.cur_time, obs
            return self.cur_time  # exit_flag = 0, still going

        elif len(self.job_full_list) > 0:  # i.e., empty cluster waiting for jobs to come
            wake_time = self.job_full_list[-1]['submit_time'] - delta  # the submit_time of the earliest job
            assert self.cur_time <= wake_time  # if ==, i.e., the stride is unnecessary
            self.cur_time = wake_time
            if return_reward:
                return self.cur_time, reward
            if type(obs) != type(None):
                return self.cur_time, obs
            return self.cur_time  # exit_flag = 0, still going

        else:  # no running job, no pending job, no coming job => exit.
            # reward = 1 # TODO: need reward engineering. All jobs done, give reward 1
            reward = 1
            if return_reward:
                return -1, reward
            if type(obs) != type(None):
                return -1, obs
            return -1  # exit

    def tic_svc(self, cur_time):
        self.cur_time = cur_time
        cap_ratio = self.get_cap_ratio(cur_time)
        svc_ratio = 1 - cap_ratio
        if self.svc_former_ratio != svc_ratio:
            self.svc_former_ratio = svc_ratio
            print_fn("%sService WAS:%s" % (self.log_prefix, str([n.__repr__() for n in self.node_list])))
            for node in self.node_list:
                if node.id in self.spare_node_id:  # spare from service allocation
                    continue
                node.set_svc_res_by_ratio(ratio=svc_ratio)
            print_fn("%sService NOW:%s" % (self.log_prefix, str([n.__repr__() for n in self.node_list])))

    def replace_svc(self):
        # Migrating services or jobs for vacancies.
        raise NotImplementedError("Cluster replace service")

    def display_capacity_pattern(self, max_time=200):
        for cur_time in range(max_time):
            cur_gpus, cur_cpus = self.get_capacity(cur_time)
            four_gpus, four_cpus = int(cur_gpus / 4), int(cur_cpus / 4)
            left_gpus, left_cpus = int(cur_gpus % 4), int(cur_cpus % 4)
            print("[%3s] G%3d |%s%s\n      C%3d |%s%s" % (cur_time, cur_gpus, "####|" * four_gpus, "#" * left_gpus, cur_cpus, "xxxx|" * four_cpus, "x" * left_cpus ))
    
    def display_capacity_pattern_csv(self, max_time=200):
        print("time,GPUs,CPUs")
        for cur_time in range(max_time):
            cur_gpus, cur_cpus = self.get_capacity(cur_time)
            # four_gpus, four_cpus = int(cur_gpus / 4), int(cur_cpus / 4)
            # left_gpus, left_cpus = int(cur_gpus % 4), int(cur_cpus % 4)
            print("%d,%d,%d" % (cur_time, cur_gpus, cur_cpus))

    def get_capacity(self, time, num_spare_node=None):
        """
        Only for display_capacity_pattern()
        :param time: cluster.cur_time, cluster.num_spare_node
        :return: [cur_gpus, cur_cpus]
        """
        num_spare_node = self.num_spare_node if num_spare_node is None else num_spare_node
        ratio = self.get_cap_ratio(time)
        if num_spare_node is None:
            return [int(ratio * self.num_gpus), int(ratio * self.num_cpus)]
        else:
            if not self.spare_node_id:
                spare_node_id = list(range(num_spare_node))
            else:
                spare_node_id = self.spare_node_id
            g, c = 0, 0
            for node in self.node_list:
                if node.id in spare_node_id:
                    g += node.num_gpus
                    c += node.num_cpus
                else:
                    g += node.num_gpus - int((1 - ratio) * node.num_gpus)
                    c += node.num_cpus - int((1 - ratio) * node.num_cpus)                    
            assert g >= 0 and c >= 0
            return [g, c]

    def get_cap_ratio(self, time, pattern=None, period=None):
        pattern = self.pattern if pattern is None else pattern
        period = self.period if period is None else period

        pattern_ratio_dict = {
            0: {1:(0,1000)}, # always maximum capacity
            1: {1:(0, 62), 0.6:(62, 124)},
            2: {0.6:[(0, 10), (62, 124)], 1:(10, 62)},
            3: {1:(0, 20), 0.9:(20, 40), 0.8:(40, 60), 0.7:(60, 80), 0.6:(80, 100), 0.5:(100, 124)},
            4: {0.5:(0, 20), 0.6:(20, 40), 0.7:(40, 60), 0.8:(60, 80), 0.9:(80, 100)},
            5: {1:[(0, 10), (110, 124)], 0.9:[(10, 20),(100, 110)], 0.8:[(20, 30),(90, 100)], 0.7:[(30, 40),(80, 90)], 0.6:(40, 50), 0.5:(50, 70), 0.4:(70, 80)},
            6: {1:[(0, 20), (50, 60), (110, 124)], 0.6:(20, 50), 0.4:(60, 110)},
            7: {1:[(0, 20), (50, 60), (110, 124)], 0.9:(20, 50), 0.8:(60, 110)}
        }  # { pattern1: {ratio1: [ (lower_bound1, upper_bound1), (lb2, ub2), ... ], ratio2: [...]},  pattern2: {...}  }

        t_mod_p = time % period
        ratio_dict = pattern_ratio_dict.get(pattern, {})
        for key, val in ratio_dict.items():
            if type(val) == tuple:
                val = [val]  # becomes a list
            for bound in val:
                if bound[0] <= t_mod_p < bound[1]:
                    return key
        return 1

    def record_cluster_util(self):
        self.cluster_time.append(self.cur_time)
        self.cluster_cpu.append(self.job_cpus)
        self.cluster_gpu.append(self.job_gpus)
    
    @property
    def node_list(self):
        return list(self.node_dict.values())

    @property
    def cur_rsrc(self):
        return [self.cur_gpus, self.cur_cpus]

    @property
    def cur_gpus(self):
        return self.num_gpus - self.svc_gpus

    @property
    def cur_cpus(self):
        return self.num_cpus - self.svc_cpus

    @property
    def job_runn_list(self):
        job_runn_list = []
        for node in self.node_list:
            job_runn_list.extend(node.job_runn_list)
        return job_runn_list

    @property
    def svc_gpus(self):
        return sum([n.svc_gpus for n in self.node_list])

    @property
    def svc_cpus(self):
        return sum([n.svc_cpus for n in self.node_list])

    @property
    def idl_gpus(self):
        return sum([n.idl_gpus for n in self.node_list])

    @property
    def idl_cpus(self):
        return sum([n.idl_cpus for n in self.node_list])

    @property
    def job_gpus(self):
        return sum([n.job_gpus for n in self.node_list])

    @property
    def job_cpus(self):
        return sum([n.job_cpus for n in self.node_list])

    @property
    def log_prefix(self):
        if self.export_cluster_util is True:  # add util export
            self.record_cluster_util()
        return "[%6s],[GPU,CPU]:[%7s,%8s]/[%7s,%8s]." % (self.cur_time, self.job_gpus, self.job_cpus, self.cur_gpus, self.cur_cpus)

