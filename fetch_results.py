import os
import json
import pandas as pd

# base_path = "/home/arda/yina/sw/LLMSuperWeight/outputs/meta-llama/Meta-Llama-3-8B-Instruct"
# base_path = "/home/arda/yina/sw/LLMSuperWeight/outputs/Qwen/Qwen2-7B-Instruct"

model_name = "Qwen2-7B-Instruct"

def walk_directory(dir_path):
    result_list = []
    for root, dirs, files in os.walk(dir_path):
        if files:
            for file in files:
                path = os.path.join(root, file)
                file_list = file.split("-")
                
                with open(path, "r") as f:
                    data = json.load(f)
                    results = data.get("results", None)
                    arc_c = results.get("arc_challenge", {}).get("acc,none", None)
                    arc_e = results.get("arc_easy", {}).get("acc,none", None)
                    lamb = results.get("lambada_openai", {}).get("acc,none", None)
                    sciq = results.get("sciq", {}).get("acc,none", None)
                    result_list.append([
                        info, "{:.4f}".format(arc_c * 100), "{:.4f}".format(arc_e * 100),
                        "{:.4f}".format(lamb * 100), "{:.4f}".format(sciq * 100)
                    ])
                print(f"root: {root}, file {file}")
            print("-" * 40)
    df = pd.DataFrame(result_list, columns=["info", "arc_c", "arc_e", "lamb", "sciq"])
    df.to_csv(f'{model_name}-results.csv', mode='w', index=False)

# Example usage
walk_directory(base_path)
