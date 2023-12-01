# csc2233_gpu_scheduling_2023Fall

Course Project for UofT CSC2233 Storage System
Code base and dataset from Alibaba trace data: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md

To run non-ML experiment, run the following command:

python cluster-trace-gpu-v2020/simulator/run_simulator.py -g 6500

,where -g specifies the number of gpus in cluster for simulation. The scheduling algorithms can be specified inside run_simulator.py with alloc_policy being defined in utils.py.



