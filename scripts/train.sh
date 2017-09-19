#!/bin/sh

python train_joined.sh --gpu 0 -i ./joined \
       -o ./models --epoch 300
