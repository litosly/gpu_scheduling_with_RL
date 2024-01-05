# csc2233_gpu_scheduling_2023Fall

Code framework of the paper "Exploring strategies to balance Fairness and Efficiency in GPU Cluster Scheduling" for University of Toronto CSC2233 Storage System course research project.

# Author Affiliate
<p align="center">
<a href="https://www.utoronto.ca//"><img src="https://github.com/k9luo/DeepCritiquingForVAEBasedRecSys/blob/master/logos/U-of-T-logo.svg" height="80"></a> 
</p>

# Dataset
Original code based for simulator and dataset we built upon: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md

# Example Experiment Commands
python cluster-trace-gpu-v2020/simulator/run_simulator.py -g 6500 -n 20000

,where -g specifies the number of gpus in cluster for simulation. The scheduling algorithms can be specified inside run_simulator.py with alloc_policy being defined in utils.py.

# Training for Reinforcement Learning
To run ML training, run the following command:
pytho cluster-trace-gpu-v2020/simulator/train_rl.py 






