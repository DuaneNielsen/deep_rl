#!/bin/bash
. /home/duane/PycharmProjects/.venv/cuda11/bin/activate
export PYTHONPATH=~/PycharmProjects/deep_rl
for i in {1..16}; do {
  wandb agent duanenielsen/reinforce-Pendulum-v0/9gge3z7z;
  } &
done