from distutils.log import info
import os
import json

# if need origin dataset dirs names
# vid_ind_list = ['1', '4', '12', '13', '14', '15', '16', '19', '20', '25', '26', '27', '29', '30', '34', '35', '38', '39', '43', '44', '50', '51', '52', '53', '59', '61', '63', '68', '70', '73', '76', '77', '78', '79', '81', '86', '91', '93']

# json_path = "./tracker/val.json"
json_path = "./wzx_val/val.json"
res_path = "./wzx_val/best_predictions.json"
# val_id_match= {}
# info_annos = json.load(open(json_path))
# for info_anno in info_annos["videos"]:
#     val_id_match["file_name"] = info_anno[info_anno['id']]
# print(val_id_match)

info_annos = json.load(open(json_path))
res_annos = json.load(open(res_path))

convert_dict = []
# print(len((info_annos["images"])))
# print(res_annos)
image_dict = {}
for info_anno in info_annos["images"]:
    image_dict[info_anno['']] = info_anno['id']
    print(info_anno['id'])
# print(image_dict)

# print(image_dict["out12_0453"])
for res in res_annos:
    # if res["score"] <= 0.002:
    #     continue
    # image_id  =res["image_id"]
    
    res["frame_id"] = image_dict[res["image_id"]][0]
    # print(image_dict)
    # exit()
    # res["video_id"] = int(vid_ind_list[image_dict[res["image_id"]][1]])   #if need origin dirs names
    res["video_id"] = int(image_dict[res["image_id"]][1])
    convert_dict.append(res)

print(len(res_annos))
print(len(convert_dict))
json.dump(convert_dict, open("./wzx_val/0713_result.json", 'w'))
