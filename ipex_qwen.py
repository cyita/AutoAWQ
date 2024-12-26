# from transformers import AutoTokenizer, AutoModelForCausalLM
# import transformers
# import torch
# import intel_extension_for_pytorch as ipex

# # Define the model ID and prompt, here we take llama2 as an example
# model_id = "/mnt/disk1/models/Qwen2-7B-Instruct"
# prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'

# # Load the tokenizer and model, move model to Intel GPU(xpu)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
# model = model.eval().to("xpu")

# print("aaaaaaa")

# # Change the memory format of the model to channels_last for optimization
# model = model.to(memory_format=torch.channels_last)

# # Optimize the model using Intel Extension for PyTorch (IPEX)
# model = ipex.llm.optimize(model.eval(), dtype=torch.float16, device="xpu")

# print("bbbbbbbb")

# # Tokenize the input prompt and convert it to tensor format and Move the input tensor to the XPU
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# input_ids = input_ids.to("xpu")

# # Generate text based on the input prompt with a maximum of 512 new tokens
# # Set cache_implementation to static cache will improve the performance, the default is dynamic cache
# # generated_ids = model.generate(input_ids, max_new_tokens=512, cache_implementation="static")[0]
# generated_ids = model.generate(input_ids, max_new_tokens=512)[0]

# # Decode the generated token IDs to a string, skipping special tokens
# generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

# # Print the generated text
# print(generated_text)

import torch

import time
import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Qwen2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="/mnt/disk1/models/Qwen2-7B-Instruct",
                        help='The Hugging Face or ModelScope repo id for the Qwen2 model to be downloaded'
                             ', or the path to the checkpoint folder')
    parser.add_argument('--prompt', type=str, default="AI是什么？",
                        help='Prompt to infer') 
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--modelscope', action="store_true", default=False, 
                        help="Use models from modelscope")

    args = parser.parse_args()
    
    model_path = args.repo_id_or_model_path


    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # import intel_extension_for_pytorch as ipex
    # model = AutoModelForCausalLM.from_pretrained(model_path,
    #                                              trust_remote_code=True,
    #                                              torch_dtype=torch.float16,
    #                                              low_cpu_mem_usage=True,
    #                                              use_cache=True)
    # model = model.eval().to('xpu')
    # model = ipex.llm.optimize(model.eval(), dtype=torch.float16, device="xpu")


    from ipex_llm.transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 optimize_model=True,
                                                 torch_dtype=torch.float16,
                                                 load_in_low_bit="fp16",
                                                 trust_remote_code=True,
                                                 use_cache=True)
    model = model.eval().to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    prompt = args.prompt

    # Generate predicted tokens
    with torch.inference_mode():
        # The following code for generation is adapted from https://huggingface.co/Qwen/Qwen2-7B-Instruct#quickstart
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
        model_inputs = tokenizer([text], return_tensors="pt").to("xpu")
        # warmup
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.n_predict
            )
        
        st = time.time()
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.n_predict
            )
        torch.xpu.synchronize()
        end = time.time()
        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(response)