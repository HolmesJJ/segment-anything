# https://github.com/facebookresearch/segment-anything/issues/54
# https://github.com/facebookresearch/segment-anything/issues/185
# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py

import os
import cv2
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator


CHECKPOINTS = ["models/sam_vit_h_4b8939.pth", "models/sam_vit_l_0b3195.pth", "models/sam_vit_b_01ec64.pth"]
MODEL_TYPES = ["vit_h", "vit_l", "vit_b"]
MODEL = 0
MIN_MASK_REGION_AREA = 50 * 50
IMAGE_SIZE = 512


RAW_DATA = glob.glob("FoodSeg103/*")
RAW_DATA_PATH = "FoodSeg103/"
DATASET_PATH = "dataset/"
DEVICE = "cuda"


def create_dataset():
    sam = sam_model_registry[MODEL_TYPES[MODEL]](checkpoint=CHECKPOINTS[MODEL])
    if torch.cuda.is_available():
        sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               pred_iou_thresh=0.95,
                                               box_nms_thresh=0.95,
                                               crop_n_layers=1,
                                               crop_nms_thresh=0.95,
                                               crop_n_points_downscale_factor=2,
                                               min_mask_region_area=MIN_MASK_REGION_AREA)
    os.makedirs(DATASET_PATH)
    with tqdm(total=len(RAW_DATA)) as pbar:
        for i, image in enumerate(RAW_DATA):
            pbar.set_description("Segmenting: %d" % (1 + i))
            pbar.update(1)
            image_name = os.path.splitext(os.path.basename(image))[0]
            image_dir = os.path.join(DATASET_PATH, image_name)
            os.makedirs(image_dir)
            image = cv2.imread(image)
            cv2.imwrite(os.path.join(DATASET_PATH, image_name + ".png"), image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            masks = [item for item in masks if item["area"] >= MIN_MASK_REGION_AREA]
            sorted_annotations = sorted(masks, key=(lambda item: item["area"]), reverse=True)
            for j, annotation in enumerate(sorted_annotations):
                img = np.zeros((sorted_annotations[0]["segmentation"].shape[0],
                                sorted_annotations[0]["segmentation"].shape[1]))
                m = annotation["segmentation"]
                img[m] = 1
                mask = cv2.bitwise_and(image, image, mask=img.astype("uint8"))
                tmp = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
                b, g, r = cv2.split(mask)
                rgba = [b, g, r, alpha]
                masked_img = cv2.merge(rgba, 4)
                contours, _ = cv2.findContours(cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                top_left_x = top_left_y = IMAGE_SIZE
                right_bottom_x = right_bottom_y = 0
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if top_left_x >= x:
                        top_left_x = x
                    if top_left_y >= y:
                        top_left_y = y
                    if right_bottom_x <= x + w:
                        right_bottom_x = x + w
                    if right_bottom_y <= y + h:
                        right_bottom_y = y + h
                cropped_img = masked_img[top_left_y:right_bottom_y, top_left_x:right_bottom_x]
                cv2.imwrite(os.path.join(image_dir, str(j) + ".png"), cropped_img)
                cv2.imwrite(os.path.join(image_dir, str(j) + "_mask.png"), mask)


def plot_image(image, show_mask=False):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    if show_mask:
        sam = sam_model_registry[MODEL_TYPES[MODEL]](checkpoint=CHECKPOINTS[MODEL])
        if torch.cuda.is_available():
            sam.to(device=DEVICE)
        mask_generator = SamAutomaticMaskGenerator(model=sam)
        masks = mask_generator.generate(image)
        show_annotations(masks)
    plt.axis("off")
    plt.show()


def show_annotations(annotations):
    if len(annotations) == 0:
        return
    sorted_annotations = sorted(annotations, key=(lambda item: item["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    annotated_image = np.ones((sorted_annotations[0]["segmentation"].shape[0],
                               sorted_annotations[0]["segmentation"].shape[1], 4))
    annotated_image[:, :, 3] = 0
    for annotation in sorted_annotations:
        m = annotation["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        annotated_image[m] = color_mask
    ax.imshow(annotated_image)


def convert_jpg_to_png():
    with tqdm(total=len(RAW_DATA)) as pbar:
        for i, image in enumerate(RAW_DATA):
            pbar.set_description("Converting: %d" % (1 + i))
            pbar.update(1)
            jpg_img = cv2.imread(image)
            jpg_fp = RAW_DATA_PATH + os.path.splitext(os.path.basename(image))[0] + ".jpg"
            png_fp = RAW_DATA_PATH + os.path.splitext(os.path.basename(image))[0] + ".png"
            cv2.imwrite(png_fp, jpg_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            os.remove(jpg_fp)


if __name__ == "__main__":
    # plot_image(RAW_DATA_PATH + "00000000.jpg", show_mask=True)
    if not os.path.isdir(DATASET_PATH):
        # convert_jpg_to_png()
        create_dataset()
