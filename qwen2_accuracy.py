#
# Copyright 2016 The BigDL Authors.
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
#
promptList = [
"给我推荐丽江三日游，包括景点和路线",
"请说出以下两句话区别在哪里？冬天能穿多少穿多少，夏天能穿多少穿多少",
"鸡兔同笼，共35只头，94只脚，问鸡兔各多少？",
"请将下面句子翻译成中文：President Joe Biden and former President Donald Trump will face each other in the U.S. presidential election on Nov. 5.",
"Python中的遍历该怎么写？","创作两句中国诗歌，关于春天和秋天。",
"如果所有猫都有尾巴，那么一只没有尾巴的动物是否可以被称为猫",
"土豆发芽了还能吃吗",
"如何评价特斯拉发布的iPhone 15？",
"我今天心情很糟糕，感觉一切都不顺利。",
"蓝牙耳机坏了需要看医院的哪个科室？",
'解释这行代码：parttern = (  \
"Intel\(R\)\s+(?:Arc\(TM\)\s+)?Graphics"   \
if inference_device == "ultra" \
else "Intel\(R\) Arc\(TM\) [^ ]+ Graphics"\
)',
"左手一只鸭，右手一只鸡。交换两次后左右手里各是什么？交换三次后左右手里各是什么？",
"以月为主题，写一首七言绝句诗",
"使用python实现读入图片，并将图片按长边尺寸resize成正方形",
"将以下中文翻译成英文：穿衣需要适应天气变化，夏天你能穿多少穿多少，冬天你能穿多少穿多少。",
"写一篇英文散文诗，主题是春雨，想象自己是春雨，和英国古代诗人莎士比亚交流",
"10.11 和 10.9 哪个大？",
"介绍一下北京烤鸭。150字左右",
"问题1：推荐三种深圳美食 \
问题2：介绍第2个",
"请将下面句子翻译成中文：The ship that my sister said that the  owner of the company claimed that the inspector had certified as seaworthy   sank in the Pacific.",
"甲、乙、丙三人比赛象棋，每两人赛一盘。胜一盘得2分。平一盘得1分，输一盘得0分。比赛的全部三盘下完后，只出现一盘平局。并且甲得3分，乙得2分，丙得1分。那么，甲乙，甲丙，乙丙（填胜、平、负）。",
"我读了一篇小小说，只有一千字，是一位上海女作家写的，后来那篇小小说得了奖。你写的这篇也一千字，你也是一位上海女作家，也一定能得奖。这个说法对吗？",
"我有两个苹果，然后我又买了两个，我用其中两个苹果烤了一个派，吃掉了一半派后，我还剩几个苹果呢？",
"请给出“海水朝朝朝朝朝朝朝落”的下联，并解释所给出下联的含义",
'用Python编程计算邮费。计算规则如下： \
根据邮件的重量和用户选择是否加急计算邮费。\
重量在1000 以内（包括），基本费8 元；\
超过1000 克的部分，每500 克加收超重费4 元，不足500克部分按500克计算；\
如果用户选择加急，多收5元。\
输入格式：\
一行，包含一个正整数x（大于1小于10e6）和一个字符c(取值为y或n)，之间用一个空格隔开，分别表示重量和是否加急。\
如果字符是   y，说明选择加急；如果字符是   n，说明不加急。\
输出格式：\
输出一行一个正整数，表示邮费',
"def create_multipliers():\
return [lambda x: i* x for i  in range(5)]\
    \
for multiplier in create_multipliers():\
print(multiplier(2))#Python中输出结果是什么",
"以月为主题，写一首诗"
]
import os
import torch
import time
import argparse

# from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# import intel_extension_for_pytorch as ipex


from transformers.utils import logging

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="The huggingface repo id for the Qwen2 or Qwen2.5 model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument("--lowbit-path", type=str,
        default="",
        help="The path to the lowbit model folder, leave blank if you do not want to save. \
            If path not exists, lowbit model will be saved there. \
            Else, lowbit model will be loaded.",
    )
    parser.add_argument('--prompt', type=str, default="AI是什么?",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=1024, help="Max tokens to predict")
    parser.add_argument("--max-context-len", type=int, default=1200)
    parser.add_argument("--max-prompt-len", type=int, default=960)
    parser.add_argument("--quantization_group_size", type=int, default=0)
    parser.add_argument('--low_bit', type=str, default="sym_int4",
                        help='Load in low bit to use')
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)
    parser.add_argument("--intra-pp", type=int, default=None)
    parser.add_argument("--inter-pp", type=int, default=None)
    parser.add_argument("--mixed-precision", action='store_true')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    # model_path = "/home/arda/llm-models/Qwen2-7B-Instruct"
    device = "cuda"


    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    # model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, trust_remote_code=True)
    # model = model.half().to("xpu")
    # model = ipex.optimize_transformers(model.eval(), dtype=torch.float16, device="xpu")

    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(r"F:\llm-models-nightly\Qwen2-7B-Instruct", trust_remote_code=True)

    if args.lowbit_path and not os.path.exists(args.lowbit_path):
        model.save_low_bit(args.lowbit_path)
    for p in promptList:
        print("-" * 80)
        print("done")
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
        with torch.inference_mode():
            print("finish to load")
            for i in range(1):
                _input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
                print("input length:", len(_input_ids[0]))
                st = time.time()
                output = model.generate(
                    _input_ids, num_beams=1, do_sample=False, max_new_tokens=args.n_predict
                )
                end = time.time()
                print(f"Inference time: {end-st} s")
                input_str = tokenizer.decode(_input_ids[0], skip_special_tokens=False)
                print("-" * 20, "Input", "-" * 20)
                print(input_str)
                output_str = tokenizer.decode(output[0], skip_special_tokens=False)
                print("-" * 20, "Output", "-" * 20)
                print(output_str)

    print("-" * 80)
    print("done")
    print("success shut down")
