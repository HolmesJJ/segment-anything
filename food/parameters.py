# https://github.com/facebookresearch/segment-anything/issues/54
# https://github.com/facebookresearch/segment-anything/issues/185
# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py

import os
import cv2
import numpy as np

from time import time
from itertools import product
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator


CHECKPOINTS = ["models/sam_vit_h_4b8939.pth", "models/sam_vit_l_0b3195.pth", "models/sam_vit_b_01ec64.pth"]
MODEL_TYPES = ["vit_h", "vit_l", "vit_b"]

IMAGE_PATHS = ["FoodSeg103/00000000.png", "FoodSeg103/00000004.png", "FoodSeg103/00000005.png",
               "FoodSeg103/00000007.png", "FoodSeg103/00000010.png"]
PARAMETERS_PATH = "parameters/"
DEVICE = "cuda"

# 5 * 5 * 2 * 5 * 2 * 5 = 2500
PARAMETERS_GRID = {
    "pred_iou_thresh": [0.75, 0.80, 0.85, 0.90, 0.95],
    "box_nms_thresh": [0.5, 0.6, 0.7, 0.8, 0.9],
    "crop_n_layers": [1, 2],
    "crop_nms_thresh": [0.5, 0.6, 0.7, 0.8, 0.9],
    "crop_n_points_downscale_factor": [1, 2],
    "min_mask_region_area": [100, 150, 200, 250, 300]
}


def generate_annotations(masks, image_name, prefix, start_time):
    if len(masks) == 0:
        return
    sorted_annotations = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    annotated_image = np.ones((sorted_annotations[0]["segmentation"].shape[0],
                               sorted_annotations[0]["segmentation"].shape[1], 4))
    annotated_image[:, :, 3] = 0
    for annotation in sorted_annotations:
        m = annotation["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        annotated_image[m] = color_mask
    annotated_image = cv2.cvtColor((annotated_image * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
    end_time = time()
    runtime = end_time - start_time
    prefix = prefix + "_time_" + str(runtime)
    cv2.imwrite(PARAMETERS_PATH + image_name + "/" + prefix + ".png", annotated_image)
    print(PARAMETERS_PATH + image_name + "/" + prefix + ".png")


def generate_mask(mask_generator, image_path, image_name, prefix, start_time):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    generate_annotations(masks, image_name, prefix, start_time)


def search_parameters_grid(image_path, image_name):
    sam = sam_model_registry[MODEL_TYPES[0]](checkpoint=CHECKPOINTS[0])
    sam.to(device=DEVICE)
    keys, values = zip(*PARAMETERS_GRID.items())
    for combination in product(*values):
        start_time = time()
        params = dict(zip(keys, combination))
        mask_generator = SamAutomaticMaskGenerator(model=sam, **params)
        prefix = "_".join([f"{key}_{value}" for key, value in params.items()])
        generate_mask(mask_generator, image_path, image_name, prefix, start_time)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.makedirs(PARAMETERS_PATH)
    for img_path in IMAGE_PATHS:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        os.makedirs(PARAMETERS_PATH + img_name)
        search_parameters_grid(img_path, img_name)
