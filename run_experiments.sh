#!/bin/bash

set -e  

# --- Question 2: Logistic Regression ---
echo "Running Question 2: Logistic Regression"
echo "Tuning learning rate for Logistic Regression with batch_size=32 and l2_decay=0.01..."

for lr in 0.00001 0.001 0.1
do
    echo "Training Logistic Regression with learning_rate=$lr..."
    python hw1-q2.py -model logistic_regression -epochs 100 -learning_rate $lr -batch_size 32 -l2_decay 0.01 -optimizer sgd
done

echo "Logistic Regression tuning completed."

# --- Question 2: Feedforward Neural Network ---
echo "Running Question 2: Feedforward Neural Network"

# Part (a): Default hyperparameters and batch_size=512
echo "Training Feedforward Neural Network with default hyperparameters..."
python hw1-q2.py -model mlp -epochs 200 -learning_rate 0.002 -hidden_size 200 -layers 2 -dropout 0.3 -batch_size 64 -activation relu -l2_decay 0.0 -optimizer sgd

echo "Training Feedforward Neural Network with batch_size=512..."
python hw1-q2.py -model mlp -epochs 200 -learning_rate 0.002 -hidden_size 200 -layers 2 -dropout 0.3 -batch_size 512 -activation relu -l2_decay 0.0 -optimizer sgd

# Part (b): Dropout experiments
echo "Running dropout experiments..."
for dropout in 0.01 0.25 0.5
do
    echo "Training Feedforward Neural Network with dropout=$dropout..."
    python hw1-q2.py -model mlp -epochs 200 -learning_rate 0.002 -hidden_size 200 -layers 2 -dropout $dropout -batch_size 64 -activation relu -l2_decay 0.0 -optimizer sgd
done

# Part (c): Momentum experiments
echo "Running momentum experiments with batch_size=1024..."
for momentum in 0.0 0.9
do
    echo "Training Feedforward Neural Network with momentum=$momentum..."
    python hw1-q2.py -model mlp -epochs 200 -learning_rate 0.002 -hidden_size 200 -layers 2 -dropout 0.3 -batch_size 1024 -activation relu -l2_decay 0.0 -optimizer sgd -momentum $momentum
done

echo "Feedforward Neural Network experiments completed."

echo "All experiments completed!"
