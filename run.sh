export LD_LIBRARY_PATH=/home/arda/miniforge3/envs/yina

# python inference_hf.py
python qwen2_accuracy.py --repo-id-or-model-path /home/arda/yina/AutoAWQ/Qwen2-7B-Instruct-awq-q40-v0 --n-predict 768 >> q40.txt
