#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 2 \
    --executor-cores 4 \
    stage3.py
