from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import itertools
from tabulate import tabulate

class_names = [ "sailing_boat",
    "fishing_boat",
    "floater",
    "passenger_ship",
    "speedboat", 
    "cargo", 
    "special_ship",
]

def per_class_mAP_table(coco_eval, class_names, headers=["class", "AP"], colums=2):
    per_class_mAP = {}
    precisions = coco_eval.eval["precision"]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_mAP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_mAP) * len(headers))
    result_pair = [x for pair in per_class_mAP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table

# coco格式的json文件，原始标注数据
anno_file = './wzx_val/val_half.json'
# res_path = './wzx_val/yolov7_tiny_0913_best.json'
res_path = './wzx_val/yolov7_tiny_0920_640_decoupled_best.json'


coco_gt = COCO(anno_file)
coco_dt = coco_gt.loadRes(res_path)

cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print(cocoEval.stats)

per_class_eval = per_class_mAP_table(cocoEval, class_names)
print(per_class_eval)
