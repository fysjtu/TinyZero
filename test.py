#模型下载
from modelscope import snapshot_download
qwen3_model_dir = snapshot_download('Qwen/Qwen2.5-1.5B')
# model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct')
print(qwen3_model_dir)
