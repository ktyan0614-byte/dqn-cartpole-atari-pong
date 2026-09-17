#!/bin/bash
# Task 2: Vanilla DQN on Atari Pong-v5 (CNN)
python atari_dqn.py \
    --save-dir ./results_task2 \
    --wandb-run-name pong-vanilla-dqn \
    --batch-size 32 \
    --memory-size 100000 \
    --lr 0.0001 \
    --discount-factor 0.99 \
    --epsilon-start 1.0 \
    --epsilon-decay 0.999995 \
    --epsilon-min 0.05 \
    --target-update-frequency 1000 \
    --replay-start-size 50000 \
    --max-episode-steps 27000 \
    --train-per-step 1
