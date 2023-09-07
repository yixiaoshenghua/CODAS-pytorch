# CODAS
The Official PyTorch Code for ["Cross-Modal Domain Adaptation for Cost-Efficient Visual Reinforcement Learning"](https://proceedings.neurips.cc/paper/2021/hash/68264bdb65b97eeae6788aa3348e553c-Abstract.html)

Tensorflow-version code: https://github.com/xionghuichen/CODAS
# Code Structure
CODAS

    |- models: code for models of CODAS
    
    |- data: the precollect dataset, pre-trained dynamics model, environments are saved here
    
    |- reset_able_mj_env: environment related code for CODAS
    
    |- configs: task configurations
    
    |- scripts: scripts to run codas
    
        |- run_data_collect.py: script to collect data of MuJoCo in the target domain
    
        |- train.py: script to train codas
    

# Quick Start
``` shell
# install python environment for CODAS
git clone --recursive https://github.com/jiangsy/mj_envs
git clone https://github.com/xionghuichen/CODAS
git clone https://github.com/jiangsy/mjrl
cd ../mj_envs/
pip install -e .
cd ../mjrl
pip install -e .
cd ../CODAS
pip install -e .

# the working directory is ./scripts
cd scripts

# run data collection in the target domain
python run_data_collect.py --env_id {task name} # to run data collect in hand DAPG envs, use the run_data_collect_robot.py script
# train codas
python train.py --env_id {task_name}
```

The training logs can be found in {your CODAS path}/log. 
