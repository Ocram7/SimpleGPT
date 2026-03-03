#!/bin/sh

thiscfg=$1

#conda activate torchtitan
export WANDB_MODE=offline

# export NCCL_SOCKET_IFNAME=front1
export NCCL_IB_DISABLE=0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# libUV is a scalable backend for TCPStore which is used in processGroup
# rendezvous. This is the recommended backend for distributed training.
export USE_LIBUV=1
# TRAINER_DIR=${TRAINER_DIR:-/home/$USER/local/torchtitan}

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run.sh

NGPU=${NGPU:-"8"}
NNODES=${NNODES:-"1"}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# by default log just rank 0 output,
LOG_RANK=${LOG_RANK:-0}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# Check if --estimate.memory=True is in the arguments
# if echo "$overrides" | grep -q -- "--memory_estimation.enabled"; then
#     # Calculate WORLD_SIZE as the product of NGPU and NNODES
#     # Export WORLD_SIZE and LOCAL_RANK
#     export WORLD_SIZE=$((NGPU * NNODES))
#     export LOCAL_RANK=0
#     python estimation.py --job.config_file ${CONFIG_FILE} $overrides
# else

# Call train.py if not in estimation mode

torchrun \
    --nproc_per_node=${NGPU} \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py --job.config_file ${thiscfg}

# fi
