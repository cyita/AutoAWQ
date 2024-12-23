from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# model_path = '/home/arda/llm-models/Qwen2-0.5B-Instruct'
# quant_path = 'Qwen2-0.5B-Instruct-awq'
# model_path = 'Qwen/Qwen2-7B-Instruct'
model_path = "/mnt/disk1/models/Qwen2-7B-Instruct"
# model_path = "/mnt/disk1/models/Qwen2-0.5B-Instruct"
quant_path = 'Qwen2-7B-Instruct-awq-q40'
# quant_config = { "zero_point": False, "q_group_size": 0, "w_bit": 4, "version": "GEMM" }
quant_config = { "zero_point": False, "q_group_size": 0, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True, use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


export_compatible = True

# Quantize
model.quantize(tokenizer, quant_config=quant_config, duo_scaling=False, export_compatible=export_compatible)
# model.quantize(tokenizer, quant_config=quant_config, export_compatible=False)

# Save quantized model
# model.save_quantized(quant_path, export_compatible=True)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
