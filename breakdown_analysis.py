import numpy as np

log_path="/home/arda/yina/AutoAWQ/breakdown_sub.log"

def get_float(line):
    return float(line[:-3])

input_features = 0
scale_search_total = 0
clip_search_total = 0

clip_search_dict = {}

scale_search_dict = {}
scale_search_sub_total_dict = {}

scale_search_flag = False
current_scale_search_name = None
scale_sub_list = []

with open(log_path, 'r') as file:
    # Read each line in the file
    for line in file:
        if "INFO" not in line:
            continue

        if scale_search_flag:
            if "Time scale" in line:
                line_list = line.split(" ")
                if line_list[2] in ["gt:", "sub:"]:
                    scale_sub_list.append(get_float(line_list[-1]))
                elif line_list[2] == "search:":
                    scale_search_sub_total_dict[current_scale_search_name] = get_float(line_list[-1])
                    scale_search_flag = False
                    scale_search_dict[current_scale_search_name] = scale_sub_list
            else:
                raise ValueError("Unexpected line in scale search")

        if "get input features" in line:
            input_features = get_float(line.split(" ")[-1])
        elif "Time search scale name" in line:
            scale_search_flag = True
            current_scale_search_name = line.split(" ")[-1][:-1]
            scale_sub_list = []
        elif "Time scale search total:" in line:
            scale_search_total = get_float(line.split(" ")[-1])
        elif "Time clip search name" in line:
            clip_list = line.split(" ")
            clip_search_dict[clip_list[-2][:-1]] = get_float(clip_list[-1])
        elif "Time clip search total" in line:
            clip_search_total = get_float(line.split(" ")[-1])

print("--------------------")
print("Get input_features = ", input_features, " ms")
print("--------------------")
print("Scale search total = ", scale_search_total, " ms\n")
inference_total = 0
sub_total = 0
for key, value in scale_search_dict.items():
    inference_total += np.sum(value)
    sub_total += scale_search_sub_total_dict[key]
    print(key, " average = ", round(np.mean(value), 4), " ms", " inference total = ", np.sum(value), " ms")
    print(key, " total = ", scale_search_sub_total_dict[key], " ms")
print("\nScale search inference total = ", inference_total, " ms")
print("\nScale search layers total = ", sub_total, " ms")
print("--------------------")
print("Clip search total = ", clip_search_total, " ms\n")
inference_total = 0
for key, value in clip_search_dict.items():
    inference_total += value
    print(key, " = ", value, " ms")
print("\nClip search inference total = ", inference_total, " ms")
