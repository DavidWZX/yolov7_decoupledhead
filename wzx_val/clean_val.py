from copy import deepcopy
import os
import json

json_path = "./wzx_val/val_half.json"

info_annos = json.load(open(json_path))

look_up = {}
for info_anno in info_annos["images"]:
    file_name = info_anno["file_name"].split('/')[2][:-4]
    look_up[info_anno["id"]] = file_name


images = deepcopy(info_annos["images"])
for i in range(len(images)):
    images[i]["file_name"] = images[i]["file_name"].split('/')[2][:-4]
    images[i]["id"] = look_up[images[i]["id"]]

# TODO
# convert_dict = {"images": images, "annotations": [], "categories": info_annos["categories"]}
# for info_anno in info_annos["annotations"]:
#     image_dict = {}
#     image_dict["id"] = info_anno["id"]
#     image_dict["image_id"] = look_up[info_anno["image_id"]]
#     image_dict["category_id"] = info_anno["category_id"] - 1
#     image_dict["bbox"] = info_anno["bbox"]
#     image_dict["iscrowd"] = info_anno["iscrowd"]
#     image_dict["area"] = info_anno["area"]
#     convert_dict["annotations"].append(image_dict)

look_up_res = {}
for info_anno in info_annos["images"]:
    file_name = info_anno["file_name"].split('/')[2][:-4]
    look_up_res[file_name] = info_anno["id"]

res_path = "./runs/test/716_ablation_0920_best_640_decoupled2/best_predictions.json"

res_annos = json.load(open(res_path))

for i in range(len(res_annos)):
    res_annos[i]["image_id"] = look_up_res[res_annos[i]["image_id"]]
    res_annos[i]["category_id"] += 1


print(len(res_annos))
json.dump(res_annos, open("./wzx_val/yolov7_tiny_0920_640_decoupled_best.json", 'w'))
