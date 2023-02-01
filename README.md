# RMA3C

Songyang Han, Sanbao Su, Sihong He, Shou Han, Haizhao Yang, and Fei Miao.

This repository implements RMA3C (Robust Multi-Agent Adversarial Actor-Critic) to learn a robust policy to maximize the average performance under worst-case state perturbations. The implementation in this repository is used in our paper "What is the Solution for State-Adversarial Multi-Agent Reinforcement Learning?"(https://arxiv.org/abs/2212.02705). We implement the RMA3C algorithm in  the folder "rma3c". This repository is based on MADDPG from https://github.com/openai/maddpg. Our baselines include MADDPG from https://github.com/openai/maddpg, M3DDPG from https://github.com/dadadidodi/m3ddpg, and MAPPO from https://github.com/marlbenchmark/on-policy.

For Multi-Agent Particle Environments (MPE) installation, we add some new scenarios to the MPE and include them here. The origin MPE repository is from https://github.com/openai/multiagent-particle-envs.

- To run the code, `cd` into the `experiments` directory of the corresponding algorithm file and run `train.py`:

``python train.py --scenario simple``

- You can replace `simple` with any environment in the MPE you'd like to run.

## Command-line options

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries`: number of adversaries in the environment (default: `0`)

- `--good-policy`: algorithm used for the 'good' (non adversary) policies in the environment
(default: `"rma3c"`; options: {`"rma3c"`, `"maddpg"`, `"ddpg"`})

- `--adv-policy`: algorithm used for the adversary policies in the environment
(default: `"rma3c"`; options: {`"rma3c"`, `"maddpg"`, `"ddpg"`})

#### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--gda-step`: number of steps to do gradient descent ascent algorithm (default: `20`)

- `--d-value`: a radius denoting how large the perturbation set is (default: `1.0`)

## Train example

### RMA3C
cd ./rma3c/experiments

python train.py --gda-step 20 --d-value 1.0  --save-dir models/ --scenario simple --exp-name simple

### MADDPG
cd ./maddpg/experiments

python train.py --save-dir models/ --scenario simple --exp-name simple

### M3DDPG
cd ./m3ddpg/experiments

python train.py --save-dir models/ --scenario simple --exp-name simple

## Citation:
If you find this repo useful in your research, please consider citing:
```bibtex
@article{han2022what,
      author = {Han, Songyang and Su, Sanbao and He, Sihong and Han, Shuo and Yang, Haizhao and Miao, Fei},
      title = {What is the Solution for State-Adversarial Multi-Agent Reinforcement Learning?},
      eprint={2212.02705},
      archivePrefix={arXiv}
      year = {2022},
}
```
