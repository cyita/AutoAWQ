from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
import torch
device = "cpu" # the device to load the model onto

# model_path = "/home/arda/yina/AutoAWQ/Qwen2-7B-Instruct-awq-asym4-4"
# # model_path = "/home/arda/llm-models/Qwen2-7B-Instruct"
# # model_path = "/home/arda/yina/AutoAWQ/Qwen2-0.5B-Instruct-awq"
# # model_path = "/home/arda/llm-models/Qwen2-0.5B-Instruct"
# model_path = "/home/arda/yina/AutoAWQ/quantized_models/Qwen2-0.5B-Instruct"
model_path = "/home/arda/yina/AutoAWQ/Qwen2-7B-Instruct-awq-q40"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)


# model = AutoAWQForCausalLM.from_quantized(
#   model_path,
# #   torch_dtype=torch.float16,
# #   low_cpu_mem_usage=True,
#   device_map="auto",
#   fuse_layers=False
# )

tokenizer = AutoTokenizer.from_pretrained(model_path)

# prompt = "Give me a short introduction to large language model."
prompt = "请将下面句子翻译成中文：The ship that my sister said that the  owner of the company claimed that the inspector had certified as seaworthy sank in the Pacific."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    do_sample=False
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
