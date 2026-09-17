#!/bin/bash
# Task 3: Enhanced DQN on Atari Pong-v5 (Double DQN + PER + Multi-step)
python enhanced_dqn.py \
    --save-dir ./results_task3 \
    --wandb-run-name pong-enhanced-dqn \
    --batch-size 32 \
    --memory-size 100000 \
    --lr 0.0000625 \
    --discount-factor 0.99 \
    --epsilon-start 1.0 \
    --epsilon-decay 0.999998 \
    --epsilon-min 0.01 \
    --target-update-frequency 16000 \
    --replay-start-size 50000 \
    --max-episode-steps 27000 \
    --train-per-step 2 \
    --n-steps 3 \
    --per-alpha 0.6 \
    --per-beta 0.4 \
    --total-steps 2500000
