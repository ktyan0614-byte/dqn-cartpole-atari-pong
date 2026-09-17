#!/bin/bash

python -m dqn \
    --save-dir ./results_task1 \
    --wandb-run-name cartpole-vanilla-dqn \
    --batch-size 64 \
    --memory-size 50000 \
    --lr 0.0005 \
    --discount-factor 0.99 \
    --epsilon-start 1.0 \
    --epsilon-decay 0.999 \
    --epsilon-min 0.05 \
    --target-update-frequency 1000 \
    --replay-start-size 1000 \
    --train-per-step 1 \
    --episodes 1500