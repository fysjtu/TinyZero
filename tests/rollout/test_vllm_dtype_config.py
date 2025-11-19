# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""确保 vLLM rollout 初始化阶段会把 dtype 设置传递到 LLM 的单元测试。"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch import nn

from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout


class _DummySamplingParams:
    """极简版 SamplingParams，用于记录传入的采样配置字段。"""

    def __init__(self, **kwargs):
        # 将所有 keyword 参数保存成属性，方便 vLLM rollout 使用 hasattr 检查。
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyLLM:
    """极简版 LLM，除了记录初始化参数外，其余接口均为空实现。"""

    def __init__(self, *_, dtype=None, **__):
        # 保存 dtype 供测试断言。
        self.dtype = dtype

    def offload_model_weights(self):
        """与真实 LLM 接口保持一致，防止 vLLM rollout 抛异常。"""

    def init_cache_engine(self):
        """测试中不会调用 generate，因此只需空实现。"""

    def free_cache_engine(self):
        """测试中不会调用 generate，因此只需空实现。"""


def test_vllm_rollout_passes_configured_dtype(monkeypatch):
    """验证 vLLM rollout 会把配置的 dtype 原样传给 LLM。"""
    # 构造满足断言的 config，其中 dtype 指定为 bfloat16。
    config = OmegaConf.create(
        {
            "tensor_model_parallel_size": 1,
            "prompt_length": 8,
            "response_length": 4,
            "dtype": "bfloat16",
            "enforce_eager": True,
            "free_cache_engine": False,
            "gpu_memory_utilization": 0.1,
            "load_format": "hf",
            "n": 1,
        }
    )

    # 覆盖 torch.distributed.get_world_size，避免在未初始化分布式环境时出错。
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)

    # 用 Dummy 对象替换掉真实 SamplingParams，隔离大依赖。
    monkeypatch.setattr(
        "verl.workers.rollout.vllm_rollout.vllm_rollout.SamplingParams",
        _DummySamplingParams,
    )

    actor = nn.Linear(8, 8)  # 轻量 Actor 模块即可满足初始化要求。
    tokenizer = SimpleNamespace(pad_token_id=0)  # 只需提供 pad_token_id。
    model_hf_config = SimpleNamespace(
        max_position_embeddings=32
    )  # 满足 context 长度断言。

    # 通过 patch 捕获 Dummy LLM 实例，便于读取 dtype。
    with patch(
        "verl.workers.rollout.vllm_rollout.vllm_rollout.LLM", wraps=_DummyLLM
    ) as mock_llm:
        vLLMRollout(actor, config, tokenizer, model_hf_config)

    # 确保 vLLM rollout 仅实例化一次 LLM，并且 dtype 与配置保持一致。
    assert mock_llm.call_count == 1, "应该只创建一次 LLM 实例"
    _, kwargs = mock_llm.call_args
    assert kwargs["dtype"] == "bfloat16", "LLM dtype 必须继承自 config.dtype"

