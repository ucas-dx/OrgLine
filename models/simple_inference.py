#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：OrgLineV1 
@File    ：simple_inference.py
@IDE     ：PyCharm 
@Author  ：Alex Deng
@Date    ：2024/7/6
'''

import onnxruntime as rt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
import requests
from tqdm import tqdm

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='#FF8585', facecolor=(0, 0, 0, 0), lw=3))

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Something went wrong during the download")

# Get current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path and URL
seg_path = os.path.join(current_script_dir, 'sam_vit_b_01ec64.pth')
url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'

# Check if the file exists, if not, download it
if not os.path.exists(seg_path):
    print(f"File not found at {seg_path}. Downloading...")
    download_file(url, seg_path)
else:
    print(f"File already exists at {seg_path}.")

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
model_path = os.path.join(current_script_dir, 'orgline.onnx')

def nms(pred, conf_thres, iou_thres):
    #print(pred.shape)
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))
    output_box = []
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]
        box_conf_sort = np.argsort(box_conf)
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)
        cls_box = np.delete(cls_box, 0, 0)
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]
                interArea = getInter(max_conf_box, current_box)
                iou = getIou(max_conf_box, current_box, interArea)
                if iou > iou_thres:
                    del_index.append(j)
            cls_box = np.delete(cls_box, del_index, 0)
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box

def getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou

def getInter(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                         box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box1[3] / 2, \
                                         box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter

def draw(img, xscale, yscale, pred, thickness=2):
    img_ = img.copy()
    abs_positions = []
    if len(pred):
        for detect in pred:
            abs_pos = [
                int((detect[0] - detect[2] / 2) * xscale),
                int((detect[1] - detect[3] / 2) * yscale),
                int((detect[0] + detect[2] / 2) * xscale),
                int((detect[1] + detect[3] / 2) * yscale)
            ]
            abs_positions.append(abs_pos)
            img_ = cv2.rectangle(img_, (abs_pos[0], abs_pos[1]), (abs_pos[2], abs_pos[3]), (0, 255, 0), thickness)
    return img_, abs_positions

def process_image(image_path, model_path,device_name ='cuda:0',conf_thres=0.5, iou_thres=0.7):
    if device_name == 'cpu':
        providers = ['CPUExecutionProvider']
    elif device_name == 'cuda:0':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    height, width = 640, 640
    img0 = cv2.imread(image_path)
    x_scale = img0.shape[1] / width
    y_scale = img0.shape[0] / height
    img = img0 / 255.
    img = cv2.resize(img, (width, height))
    img = np.transpose(img, (2, 0, 1))
    data = np.expand_dims(img, axis=0)
    sess = rt.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    result = nms(pred, conf_thres=conf_thres, iou_thres=iou_thres)
    ret_img, abs_positions = draw(img0, x_scale, y_scale, result)
    ret_img = ret_img[:, :, ::-1]
    return {os.path.abspath(image_path): abs_positions}

class OrgAnalysis:
    def __init__(self, image_folder, model_path=model_path, device_name='cpu', conf_thres=0.5, iou_thres=0.7):
        self.image_folder = image_folder
        self.model_path = model_path
        self.device_name = device_name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def analyze(self, show_bboxes=True,show_seg=False, num_images_per_row=4):
        image_paths = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, image_path, self.model_path, self.device_name, self.conf_thres, self.iou_thres) for image_path in image_paths]
            results = [future.result() for future in futures]

        combined_results = {}
        for result in results:
            combined_results.update(result)

        #print(combined_results)

        if show_bboxes:
            num_rows = len(image_paths) // num_images_per_row + (len(image_paths) % num_images_per_row > 0)
            plt.figure(figsize=(20, num_rows * 5), dpi=300)  

            for idx, image_path in enumerate(image_paths):
                img_abs_path = os.path.abspath(image_path)
                result_img = cv2.imread(img_abs_path)
                result_positions = combined_results[img_abs_path]

                for pos in result_positions:
                    result_img = cv2.rectangle(result_img, (pos[0], pos[1]), (pos[2], pos[3]), (255, 255, 0), 2)

                plt.subplot(num_rows, num_images_per_row, idx + 1)
                plt.imshow(result_img[:, :, ::-1])
                plt.title(f"{os.path.basename(img_abs_path)}")
            plt.tight_layout()
            plt.show()

            # Create a new figure for analysis plots
            plt.figure(figsize=(20, 5))

            width_height_data = []
            organ_counts = []
            image_labels = []

            # Define a color map
            cmap = plt.get_cmap("Set1")

            for idx, image_path in enumerate(image_paths):
                img_abs_path = os.path.abspath(image_path)
                result_positions = combined_results[img_abs_path]

                # Calculate image width and height
                img = cv2.imread(img_abs_path)
                img_height, img_width, _ = img.shape

                for pos in result_positions:
                    x1, y1, x2, y2 = pos
                    width = x2 - x1
                    height = y2 - y1

                    # Calculate relative sizes
                    rel_width = width / img_width
                    rel_height = height / img_height

                    width_height_data.append((os.path.basename(img_abs_path), rel_width, rel_height))
                organ_counts.append(len(result_positions))
                image_labels.append(os.path.basename(img_abs_path))

                # Plot width-height distribution
            plt.subplot(1, 2, 1)
            for idx, label in enumerate(image_labels):
                data = [(w, h) for l, w, h in width_height_data if l == label]
                if data:
                    widths, heights = zip(*data)
                    sizes = [1 for _ in data]  # Set a constant size for all points
                    plt.scatter(widths, heights, s=50, label=label, color=cmap(idx % 10))
            plt.xlabel('Relative Width')
            plt.ylabel('Relative Height')
            plt.title('Relative Width vs Height Distribution')
            plt.xlim(0, 0.5)
            plt.ylim(0, 0.5)
            plt.legend()

            # Plot organ count distribution
            plt.subplot(1, 2, 2)
            plt.bar(image_labels, organ_counts, color=[cmap(i % 10) for i in range(len(image_labels))])
            plt.xlabel('Image')
            plt.ylabel('Organoids Count')
            plt.title('Organoids Count per Image')

            plt.tight_layout()
            plt.show()
        if show_seg:
            import sys
            import torch
            sys.path.append("..")
            from models.segment_anything import sam_model_registry, SamPredictor
            sam_checkpoint = seg_path
            model_type = "vit_b"
            device = "cuda"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            out_mask = []
            num_rows = len(image_paths) // num_images_per_row + (len(image_paths) % num_images_per_row > 0)
            fig, axs = plt.subplots(num_rows, num_images_per_row, figsize=(20, num_rows * 5), dpi=300)
            axs = axs.flatten()  # Flatten the 2D array of axes
            plt.figure(figsize=(20, num_rows * 5), dpi=300)  

            for idx, image_file in enumerate(image_paths):  
                image_file = os.path.abspath(image_file)
                #print(image_file)
                img = cv2.imread(image_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_boxes = []

                for line in combined_results.get(image_file, []):
                    label_data = line
                    input_boxes.append(torch.tensor(label_data, device=device))

                if input_boxes:
                    input_boxes = torch.stack(input_boxes, dim=0)
                    predictor.set_image(img)
                    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])
                    masks, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )
                    ax = axs[idx]
                    ax.imshow(img)
                    for mask in masks:
                        show_mask(mask.cpu().numpy(), ax, random_color=True)
                    #ax.axis('off')
                    ax.set_title(os.path.basename(image_file))

            for i in range(idx + 1, len(axs)):  # Remove empty subplots
                fig.delaxes(axs[i])
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    image_folder = '../images'

    org_analysis = OrgAnalysis(image_folder)
    org_analysis.analyze(show_bboxes=False,show_seg=True)




