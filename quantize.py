
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# set HF_ENDPOINT=https://hf-mirror.com

model_path = r"F:\llm-models-nightly\Qwen2-7B-Instruct"
quant_path = 'Qwen2-7B-Instruct-awq-q40-woduo-test2'
# quant_config = { "zero_point": False, "q_group_size": 0, "w_bit": 4, "version": "GEMM" }
quant_config = { "zero_point": False, "q_group_size": 0, "w_bit": 4, "version": "GEMM" }

# # Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True, use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


export_compatible = True

# Quantize
model.quantize(tokenizer, quant_config=quant_config, duo_scaling=False, export_compatible=export_compatible)
# model.quantize(tokenizer, quant_config=quant_config, export_compatible=False)

if model.model.lm_head.weight_type == 2:
    print("transpose lmhead.....")
    model.model.lm_head.weight = torch.nn.Parameter(model.model.lm_head.weight.transpose(0, 1).contiguous(),
                                                     requires_grad=False)
    model.model.lm_head.weight_type = 1


# Save quantized model
model.save_quantized(quant_path, quantized=False)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
