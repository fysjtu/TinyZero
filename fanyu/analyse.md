### 总体概览

- 技术栈与依赖
  - Python 包名称为 `verl`（见 `pyproject.toml`），依赖 `transformers(<4.48)`、`vllm(<=0.6.3)`、`ray`、`hydra-core`、`torch`、`datasets`、`tensordict` 等。
  - 强化学习训练基于 veRL 框架，使用 Ray 做单机/多机的进程与资源编排，训练算法为 PPO/GRPO 变体。
- 主要能力
  - 通过混合引擎（hybrid engine）组织 Actor/Rollout、Critic、RefPolicy、RewardModel 等角色，并支持 FSDP 与 Megatron 两种并行后端。
  - 数据与奖励:
    - 训练数据来自 parquet 文件，数据管线内置 prompt 截断、collate 等。
    - 奖励支持规则函数（如 gsm8k、countdown 任务）与模型式 RM 组合；支持 KL 惩罚控制。
- 运行入口
  - 根 `README.md` 给出环境搭建与训练脚本调用（`scripts/train_tiny_zero.sh`）；核心训练入口为 `verl/trainer/main_ppo.py`。

### 目录结构（高频路径）

- `verl/` 核心包
  - `trainer/`：训练主逻辑与配置
    - `main_ppo.py`：Hydra 配置入口，创建 Ray 资源池、角色与训练器
    - `ppo/ray_trainer.py`：Ray 版本 PPO 训练循环与数据流
    - `config/*.yaml`：训练配置（PPO 参数、数据路径、并行策略等）
  - `single_controller/`：抽象 Worker、WorkerGroup 与后端实现
    - `base/worker.py`、`base/worker_group.py`：Worker 基类、分发与方法绑定机制
    - `ray/`：Ray 后端 WorkerGroup/ResourcePool 实现
    - `megatron/`：Megatron 相关
  - `workers/`：具体角色的 Worker 实现
    - `fsdp_workers.py`、`megatron_workers.py` 等
  - `utils/`：数据集、tokenizer、奖励函数、日志、均衡分配等工具
  - `models/`：模型注册、加载与权重处理
- `examples/`：数据预处理与训练示例脚本
- `tests/`：单元/集成测试，包含 GPU utility、model、ray、rollout 等测试
- `docs/`：Sphinx 文档（安装、Quickstart、PPO 代码架构等）

### 训练主入口与奖励组合

- `verl/trainer/main_ppo.py` 是训练入口，使用 Hydra 配置并在 Ray 远程任务中初始化 Tokenizer、角色映射、资源池、奖励函数组合，最后启动 `RayPPOTrainer`。

代码参考（入口与奖励选择）:

```92:120:verl/trainer/main_ppo.py
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))
```

```24:35:verl/trainer/main_ppo.py
def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    else:
        raise NotImplementedError
```

```164:190:verl/trainer/main_ppo.py
if config.reward_model.enable:
    ...
reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

trainer = RayPPOTrainer(config=config,
                        tokenizer=tokenizer,
                        role_worker_mapping=role_worker_mapping,
                        resource_pool_manager=resource_pool_manager,
                        ray_worker_group_cls=ray_worker_group_cls,
                        reward_fn=reward_fn,
                        val_reward_fn=val_reward_fn)
trainer.init_workers()
trainer.fit()
```

- `RewardManager` 会针对不同数据源选择不同的规则打分函数（或直接使用 batch 内提供的 rm_scores）。

代码参考（RewardManager 计算并可打印样例）:

```41:90:verl/trainer/main_ppo.py
class RewardManager():
    ...
    def __call__(self, data: DataProto):
        ...
        for i in range(len(data)):
            ...
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score
            ...
        return reward_tensor
```

### RayPPOTrainer：数据流与训练循环

- 架构职责
  - 初始化资源池与 WorkerGroup（不同角色使用同一或不同池，支持共置）。
  - 创建 DataLoader（训练与验证），注入 `total_training_steps`。
  - 训练循环中完成：生成、RM/规则评分、KL 惩罚、优势计算、Critic/Actor 更新、验证与 checkpoint。
- 关键角色
  - `Role.ActorRollout`：负责生成与 Actor 更新；`RefPolicy` 用于 KL 惩罚的参考分布；`Critic` 估计 value；`RewardModel` 提供模型式奖励。
- 重要步骤
  1) 使用 `actor_rollout_wg.generate_sequences` 生成回复
  2) 对 batch 做序列长度均衡切分，减少 DP 不平衡
  3) 计算 ref policy 的 log_prob（若启用）
  4) 计算 values（若使用 GAE）
  5) 组合奖励（RM + 规则），并加 KL 惩罚
  6) 计算优势（GAE 或 GRPO）
  7) 更新 Critic 与 Actor，定期验证与保存

代码参考（初始化与循环关键节点）:

```340:391:verl/trainer/ppo/ray_trainer.py
def _create_dataloader(self):
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files, ...)
    self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.config.data.train_batch_size, ...)
    ...
    total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
    if self.config.trainer.total_training_steps is not None:
        total_training_steps = self.config.trainer.total_training_steps
    ...
    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
    self.config.critic.optim.total_training_steps = total_training_steps
```

```444:515:verl/trainer/ppo/ray_trainer.py
def init_workers(self):
    self.resource_pool_manager.create_resource_pool()
    ...
    # actor+rollout 共置
    actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                             config=self.config.actor_rollout_ref, role='actor_rollout')
    ...
    if self.config.algorithm.adv_estimator == 'gae':
        critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
        self.use_critic = True
    elif self.config.algorithm.adv_estimator == 'grpo':
        self.use_critic = False
    ...
    if self.use_reference_policy:
        ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role='ref')
    ...
    # WorkerGroup 启动并 init_model
    wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    ...
    self.actor_rollout_wg = all_wg['actor_rollout']
    self.actor_rollout_wg.init_model()
```

```575:606:verl/trainer/ppo/ray_trainer.py
for epoch in range(self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        ...
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        ...
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)
        self._balance_batch(batch, metrics=metrics)
        ...
        if self.use_reference_policy:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)
        if self.use_critic:
            values = self.critic_wg.compute_values(batch)
            batch = batch.union(values)
```

```618:645:verl/trainer/ppo/ray_trainer.py
if self.use_rm:
    reward_tensor = self.rm_wg.compute_rm_score(batch)
    batch = batch.union(reward_tensor)
reward_tensor = self.reward_fn(batch)
batch.batch['token_level_scores'] = reward_tensor

if not self.config.actor_rollout_ref.actor.use_kl_loss:
    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty)
    metrics.update(kl_metrics)
else:
    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, ...)
```

```647:671:verl/trainer/ppo/ray_trainer.py
if self.use_critic:
    critic_output = self.critic_wg.update_critic(batch)
...
if self.config.trainer.critic_warmup <= self.global_steps:
    actor_output = self.actor_rollout_wg.update_actor(batch)
...
if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
    val_metrics: dict = self._validate()
...
if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
    self._save_checkpoint()
```

### 单机控制器与 Worker 基类

- `Worker` 负责从环境变量中解析分布式信息（WORLD_SIZE/RANK 等），实现元信息注入与方法注册装饰器的通用执行路径。
- `WorkerGroup` 提供方法分发、收集、绑定，将用户定义的带注解方法映射到分发策略（如 all-to-all、rank_zero）与执行模式。

代码参考（Worker 环境元信息注入）:

```119:161:verl/single_controller/base/worker.py
def __init__(self, cuda_visible_devices=None) -> None:
    import os
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    ...
    meta = WorkerMeta(store=store)
    self._configure_with_meta(meta=meta)

def _configure_with_meta(self, meta: WorkerMeta):
    ...
    for key in WorkerMeta.keys:
        val = self.__dict__.get(f"_{key.lower()}", None)
        if val is not None:
            os.environ[key] = str(val)
    os.environ["REDIS_STORE_SERVER_HOST"] = str(self._master_addr).replace("[", "").replace("]", "") if self._master_addr else ""
```

代码参考（WorkerGroup 绑定注册方法到分发执行）:

```136:197:verl/single_controller/base/worker_group.py
def _bind_worker_method(self, user_defined_cls, func_generator):
    for method_name in dir(user_defined_cls):
        ...
        if hasattr(method, MAGIC_ATTR):
            attribute = getattr(method, MAGIC_ATTR)
            ...
            if isinstance(dispatch_mode, Dispatch):
                fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                dispatch_fn = fn['dispatch_fn']
                collect_fn = fn['collect_fn']
            ...
            execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
            wg_execute_fn_name = execute_mode['execute_fn_name']
            execute_fn = getattr(self, wg_execute_fn_name)
            ...
            func = func_generator(self, method_name, dispatch_fn=dispatch_fn, collect_fn=collect_fn, execute_fn=execute_fn, blocking=blocking)
            setattr(self, method_name, func)
```

### 配置与运行

- 配置位于 `verl/trainer/config/ppo_trainer.yaml`（未展开，但从入口与 trainer 逻辑可知），主要包含：
  - `data.*`：训练/验证 parquet 文件、batch size、prompt key、长度限制等
  - `trainer.*`：epoch、步数、保存/验证频率、logger、项目名/实验名等
  - `actor_rollout_ref.*`：模型路径、并行配置、rollout 重复、是否使用 KL loss 等
  - `critic.*` 与 `algorithm.*`：优化器、GAE/GRPO、gamma/lam、KL 控制器等
- 运行方式
  - 根 README 中提供 conda/pip 安装步骤与示例脚本；核心是设置环境变量后执行 `scripts/train_tiny_zero.sh`。

代码参考（包元数据及依赖）:

```31:44:pyproject.toml
dependencies = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy",
    "pybind11",
    "ray",
    "tensordict",
    "transformers<4.48",
    "vllm<=0.6.3",
]
```

### 测试结构

- `tests/` 包含多层面的测试：
  - `gpu_utility/`：内存缓冲与算子功能测试
  - `model/`：transformer 与 Ulysses 并行测试
  - `ray/`：Worker 生存、数据传输、高层调度 API、本地环境等
  - `rollout/`：vLLM 相关加载与运行
  - `utility/`：张量字典工具测试
  - `e2e/`：端到端场景（如 arithmetic_sequence）

### 你可以从哪里改起

- 新任务/奖励类型：扩展 `verl/utils/reward_score/*` 与 `_select_rm_score_fn` 的映射；或增加模型式 RM 并在 `config.reward_model` 打开。
- 算法变体：在 `verl/trainer/ppo/core_algos`（未展开）添加新的优势估计或 KL 控制策略，在 `ray_trainer.py` 的分支接入。
- 并行策略：切换 `actor_rollout_ref.actor.strategy` 为 `fsdp`/`megatron`，对应 `verl/workers/fsdp_workers.py` 与 `verl/workers/megatron_workers.py`。
- 数据管线：修改 `verl/utils/dataset/rl_dataset.py` 的 collate 或过滤策略。

### 单元测试建议（覆盖关键新改动）

- 针对新奖励函数：
  - 构造最小 `DataProto` 样例，验证 `RewardManager` 对该数据源返回的 `reward_tensor` 形状与位置（最后一个有效 token）正确。
- 针对新优势估计/KL 控制：
  - 以小 batch 构造固定 `token_level_scores`、`attention_mask`、`old_log_probs/ref_log_prob`，断言 `advantages/returns/kl` 数值与预期公式一致。
- 针对 Worker 调度扩展：
  - 在本地 Ray 环境中启动最小世界（1-2 进程），验证 `generate_sequences`、`compute_values` 的 RPC 调用是否返回预期 `DataProto` 字段。
- 针对数据均衡：
  - 构造不同长度样本，调用 `_balance_batch` 后检查各 DP 分区 token 总数差异是否降低。

如果你希望，我可以继续深入某个具体子模块（例如 `fsdp_workers` 内部、`dataset` 实现、或 `core_algos` 公式）做更细的逐文件讲解与内联注释版本。

- 我已完成初步框架分析，接下来如果你指定模块，我会按“逐文件更改”的方式深入并补充详细注释与相应测试建议。
