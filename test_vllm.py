from vllm import LLM, SamplingParams

llm = LLM(
    model="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B",
    dtype="bfloat16",
    tensor_parallel_size=2,
    enforce_eager=True,
    enable_chunked_prefill=False,   # <--
    # attention_backend="FLASH_ATTN", # 如无 cudagraph 需求，推荐 Flash-Attn
)
print(llm.generate(["hello"], SamplingParams(max_tokens=8)))
