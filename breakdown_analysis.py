import numpy as np

log_path="/home/arda/yina/AutoAWQ/q40-wo-duo.log"

def get_float(line):
    return float(line[:-3])

input_features = 0
scale_search_total = 0
clip_search_total = 0

clip_search_dict = {}
clip_search_sub_total_dict = {}
clip_sub_list = []
clip_pseudo_list = []
clip_gt_list = []

scale_search_dict = {}
scale_search_sub_total_dict = {}

scale_search_flag = False
current_scale_search_name = None
scale_sub_list = []
scale_pseudo_list = []
scale_loss_list = []
scale_gt_list = []

def print_results():
    global input_features, scale_search_total, clip_search_total, clip_search_dict, clip_search_sub_total_dict, clip_sub_list, clip_pseudo_list, clip_gt_list, scale_search_dict, scale_search_sub_total_dict, scale_search_flag, current_scale_search_name, scale_sub_list, scale_pseudo_list, scale_loss_list, scale_gt_list
    print("--------------------")
    print("Get input_features = ", input_features, " ms")
    print("--------------------")
    print("Scale search total = ", scale_search_total, " ms\n")
    inference_total = 0
    scale_sub_total = 0
    pseudo_total = 0
    loss_total = 0
    for key, value in scale_search_dict.items():
        sub_infer_total = (np.sum(value['sub']) + value['gt'][0])
        inference_total += sub_infer_total
        scale_gt = (scale_search_sub_total_dict[key] + value['gt'][0])
        scale_sub_total += scale_gt
        gt_inf_pseudo_loss_total = np.sum(value['sub']) + value['gt'][0] + np.sum(value['pseudo']) + np.sum(value['loss'])
        pseudo_total += np.sum(value['pseudo'])
        loss_total += np.sum(value['loss'])
        print(key, " gt inference = ", value['gt'][0], " ms", " search grid average = ", round(np.mean(value['sub']), 4), " ms", " inference total = ", round(sub_infer_total, 4), " ms")
        print(key, " pseudo average= ", round(np.mean(value['pseudo']), 4), " ms", " pseudo total = ", round(np.sum(value['pseudo']), 4), " ms")
        print(key, " loss average= ", round(np.mean(value['loss']), 4), " ms", " loss total = ", round(np.sum(value['loss']), 4), " ms")
        print(key, " inference + pseudo + loss = ", round(gt_inf_pseudo_loss_total, 4), " ms")
        print(key, " total = ", scale_gt, " ms\n")
    print("\nScale search inference total = ", round(inference_total / 1000, 4), " s")
    print("Scale search pseudo total = ", round(pseudo_total / 1000, 4), " s")
    print("Scale search loss total = ", round(loss_total / 1000, 4), " s")
    print("\nScale search layers total = ", round(scale_sub_total /1000, 4), " s")
    print("--------------------")

    inference_total = 0
    sub_total = 0
    pseudo_total = 0
    for key, value in clip_search_dict.items():
        sub_total += clip_search_sub_total_dict[key]
        sub_infer_total = (np.sum(value['sub']) + np.sum(value['gt']))
        inference_total += sub_infer_total
        pseudo_total += np.sum(value['pseudo'])
        print(key, " gt inference average = ", round(np.mean(value['gt']), 4), " ms", " search grid average = ", round(np.mean(value['sub']), 4), " ms", " inference total = ", round(sub_infer_total, 4), " ms")
        print(key, " pseudo average = ", round(np.mean(value['pseudo']), 4), " ms", " pseudo total = ", round(np.sum(value['pseudo']), 4), " ms")
        
        print(key, " = ", clip_search_sub_total_dict[key], " ms")
    print("\nClip search inference total = ", round(inference_total / 1000, 4), " s")
    print("Clip search pseudo total = ", round(pseudo_total / 1000, 4), " s")
    print("\nClip search layers total = ", round(sub_total /1000, 4), " ms")
    print("Clip search total = ", round(clip_search_total /1000, 4), " s\n")

    print("--------------------")
    print("Scale + clip total = ", round((scale_sub_total + clip_search_total) / 1000, 4), " s")

    input_features = 0
    scale_search_total = 0
    clip_search_total = 0

    clip_search_dict = {}
    clip_search_sub_total_dict = {}
    clip_sub_list = []
    clip_pseudo_list = []
    clip_gt_list = []

    scale_search_dict = {}
    scale_search_sub_total_dict = {}

    scale_search_flag = False
    current_scale_search_name = None
    scale_sub_list = []
    scale_pseudo_list = []
    scale_loss_list = []
    scale_gt_list = []


with open(log_path, 'r') as file:
    # Read each line in the file
    for line in file:
        if "INFO" not in line:
            continue

        if "Module: " in line:
            idx = line.split(" ")[-1][:-1]
            if idx != "0":
                print_results()
            print("\n\n")
            print("***************** Module: ", idx, " *****************")

        if scale_search_flag:
            if "Time scale" in line:
                line_list = line.split(" ")
                if line_list[7] == "sub:":
                    scale_sub_list.append(get_float(line_list[-1]))
                elif line_list[7] == "gt:":
                    scale_gt_list.append(get_float(line_list[-1]))
                elif line_list[7] == "pseudo:":
                    scale_pseudo_list.append(get_float(line_list[-1]))
                elif line_list[7] == "loss:":
                    scale_loss_list.append(get_float(line_list[-1]))
                elif line_list[7] == "search:":
                    scale_search_sub_total_dict[current_scale_search_name] = get_float(line_list[-1])
                    scale_search_flag = False
                    scale_search_dict[current_scale_search_name] = {
                        "sub": scale_sub_list,
                        "gt": scale_gt_list,
                        "pseudo": scale_pseudo_list,
                        "loss": scale_loss_list
                    }
            else:
                raise ValueError("Unexpected line in scale search")

        if "get input features" in line:
            input_features = get_float(line.split(" ")[-1])
        elif "Time search scale name" in line:
            scale_search_flag = True
            current_scale_search_name = line.split(" ")[-1][:-1]
            scale_sub_list = []
            scale_pseudo_list = []
            scale_loss_list = []
            scale_gt_list = []
        elif "Time scale search total:" in line:
            scale_search_total = get_float(line.split(" ")[-1])
        elif "Time clip search name" in line:
            clip_list = line.split(" ")
            clip_search_sub_total_dict[clip_list[-2][:-1]] = get_float(clip_list[-1])
            clip_search_dict[clip_list[-2][:-1]] = {
                "sub": clip_sub_list,
                "pseudo": clip_pseudo_list,
                "gt": clip_gt_list
            }
            clip_sub_list = []
            clip_pseudo_list = []
            clip_gt_list = []
        elif "Time clip search total" in line:
            clip_search_total = get_float(line.split(" ")[-1])
        elif "Time clip search gt:" in line:
            clip_gt_list.append(get_float(line.split(" ")[-1]))
        elif "Time clip search pseudo:" in line:
            clip_pseudo_list.append(get_float(line.split(" ")[-1]))
        elif "Time clip search sub:" in line:
            clip_sub_list.append(get_float(line.split(" ")[-1]))
        
print_results()
# print("--------------------")
# print("Get input_features = ", input_features, " ms")
# print("--------------------")
# print("Scale search total = ", scale_search_total, " ms\n")
# inference_total = 0
# sub_total = 0
# pseudo_total = 0
# loss_total = 0
# for key, value in scale_search_dict.items():
#     sub_infer_total = (np.sum(value['sub']) + value['gt'][0])
#     inference_total += sub_infer_total
#     scale_gt = (scale_search_sub_total_dict[key] + value['gt'][0])
#     sub_total += scale_gt
#     gt_inf_pseudo_loss_total = np.sum(value['sub']) + value['gt'][0] + np.sum(value['pseudo']) + np.sum(value['loss'])
#     pseudo_total += np.sum(value['pseudo'])
#     loss_total += np.sum(value['loss'])
#     print(key, " gt inference = ", value['gt'][0], " ms", " search grid average = ", round(np.mean(value['sub']), 4), " ms", " inference total = ", round(sub_infer_total, 4), " ms")
#     print(key, " pseudo average= ", round(np.mean(value['pseudo']), 4), " ms", " pseudo total = ", round(np.sum(value['pseudo']), 4), " ms")
#     print(key, " loss average= ", round(np.mean(value['loss']), 4), " ms", " loss total = ", round(np.sum(value['loss']), 4), " ms")
#     print(key, " inference + pseudo + loss = ", round(gt_inf_pseudo_loss_total, 4), " ms")
#     print(key, " total = ", scale_gt, " ms\n")
# print("\nScale search inference total = ", round(inference_total / 1000, 4), " s")
# print("Scale search pseudo total = ", round(pseudo_total / 1000, 4), " s")
# print("Scale search loss total = ", round(loss_total / 1000, 4), " s")
# print("\nScale search layers total = ", round(sub_total /1000, 4), " s")
# print("--------------------")

# inference_total = 0
# sub_total = 0
# pseudo_total = 0
# for key, value in clip_search_dict.items():
#     sub_total += clip_search_sub_total_dict[key]
#     sub_infer_total = (np.sum(value['sub']) + np.sum(value['gt']))
#     inference_total += sub_infer_total
#     pseudo_total += np.sum(value['pseudo'])
#     print(key, " gt inference average = ", round(np.mean(value['gt']), 4), " ms", " search grid average = ", round(np.mean(value['sub']), 4), " ms", " inference total = ", round(sub_infer_total, 4), " ms")
#     print(key, " pseudo average = ", round(np.mean(value['pseudo']), 4), " ms", " pseudo total = ", round(np.sum(value['pseudo']), 4), " ms")
    
#     print(key, " = ", clip_search_sub_total_dict[key], " ms")
# print("\nClip search inference total = ", round(inference_total / 1000, 4), " s")
# print("Clip search pseudo total = ", round(pseudo_total / 1000, 4), " s")
# print("\nClip search layers total = ", round(sub_total /1000, 4), " ms")
# print("Clip search total = ", round(clip_search_total /1000, 4), " s\n")