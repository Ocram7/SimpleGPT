# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama import (
    llama2_configs,
    llama3_configs,
    PrenormTransformer,
    SimpleGPT,
    PreQKNormTransformer,
)

models_config = {
    "llama2": llama2_configs,
    "llama2_simplegpt": llama2_configs,
    "llama2_preqknorm": llama2_configs,
    "llama3": llama3_configs,
    "llama3_simplegpt": llama3_configs,
    "llama3_preqknorm": llama3_configs,
}

model_name_to_cls = {
    "llama2": PrenormTransformer,
    "llama2_simplegpt": SimpleGPT,
    "llama2_preqknorm": PreQKNormTransformer,
    "llama3": PrenormTransformer,
    "llama3_simplegpt": SimpleGPT,
    "llama3_preqknorm": PreQKNormTransformer,
}

model_name_to_tokenizer = {
    "llama2": "sentencepiece",
    "llama2_simplegpt": "sentencepiece",
    "llama2_preqknorm": "sentencepiece",
    "llama3": "tiktoken",
    "llama3_simplegpt": "tiktoken",
    "llama3_preqknorm": "tiktoken",
}
