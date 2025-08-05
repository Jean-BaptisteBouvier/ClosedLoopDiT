# Closed-loop DiT

Diffusion planning in closed-loop for robot locomotion.
Building on the [DDAT](https://iconlab.negarmehr.com/DDAT/) project which operates only in open-loop.
All simulations rely on the MuJoCo physics engine.

## Hopper

The [Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) is the simplest dynamical system on which we work.

This Hopper branch contains all the code necessary to train a single-step RL policy with PPO.
Make a dataset of trajectories.
Train and evaluate a Diffusion Transformer (DiT) on this dataset.

The first objective is to generate closed-loop trajectories of the Hopper with DiT.


## Reading list

- Introduction to Reinforcement Learning (RL) with the [OpenAI blog](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- Proximal Policy Optimization (PPO) [blog](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) [paper](https://arxiv.org/abs/1707.06347)
- PyTorch [tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)


- Introduction to diffusion models [blog](https://theaisummer.com/diffusion-models/)
- Foundational paper for diffusion models [Denoising Diffusion Probabilistic Model (DDPM)](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
- Formalizing the diffusion models [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf)
- Hands-on introduction to diffusion models, with simple codes for 1D and 2D models you can understand and run [blog](https://jean-baptistebouvier.github.io/assets/blog_posts/diffusion/)
- Using diffusion models to generate robot trajectories: [Diffuser](https://arxiv.org/pdf/2205.09991)
- This repo builds on our work making diffusion models generate feasible robot trajectories [DDAT](https://iconlab.negarmehr.com/DDAT/)


## Setup

Create a conda environment `hopper-env` with all the necessary packages specified in `environment.yml`.

```
conda env create -f environment.yml
```

Clone this branch of the repo on your computer.



## Code organization

- `datasets` contains datasets of trajectories generated with `dataset_making.py` from PPO policies saved in the folder `policies`.
- `policies` contains the PPO policies trained with `train_ppo.py`.
- `trained` contains the DiT models trained with `train_DiT.py` and ready to be evaluated with `eval_DiT.py`.
- `DiT_SA.py` codes the Diffusion Transformer (DiT) to generate State-Action (SA) trajectories.
- `PPO.py` is the PPO implementation modified from this [repo](https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous)


## TODO list

- [ ] create conda environment
- [ ] clone this branch on your machine
- [ ] familiarize yourself with the Hopper environment `hopper.py`
- [ ] run `train_ppo.py` to train a PPO policy
- [ ] make a dataset of trajectories with your PPO policy by running `dataset_making.py`
- [ ] train a DiT on your dataset with `train_DiT.py`
- [ ] evaluate your DiT with `eval_DiT.py`

