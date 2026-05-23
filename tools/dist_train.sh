#!/usr/bin/env bash

#CONFIG=$1
#CHECKPOINT=$2
#GPUS=$3
##PORT=${PORT:-28509}
#PORT=${PORT:-28508}
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --resume_from $CHECKPOINT --launcher pytorch ${@:4} --deterministic
##python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
##    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic

CONFIG=$1
GPUS=$2
PORT=${PORT:-28508}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic

#CONFIG=$1
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python $(dirname "$0")/train.py $CONFIG --launcher none --deterministic
