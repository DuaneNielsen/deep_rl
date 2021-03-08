#!/bin/bash
. /home/duane/PycharmProjects/.venv/cuda11/bin/activate
export PYTHONPATH=~/PycharmProjects/deep_rl
for i in {1..10}; do {
  wandb agent duanenielsen/reinforce-CartPoleContinuous-v1/s29tbywl;
  } &
done