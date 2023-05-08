import os
import torch
import cv2
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

source = './images/1.jpg'
evaluate = True
weights = './train/weights/best.pt'
imgsz=(640, 640)  # inference size (height, width)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS

def predict_results():
    model = torch.hub.load('..', 'custom', path=weights, device='cpu', source='local')
    model.conf = conf_thres # NMS confidence threshold
    model.iou = iou_thres  # NMS IoU threshold

    img = cv2.imread(source)
    results = model(img,size=imgsz)

    results_pandas = results.pandas().xyxy[0]
    results_dict = results_pandas.set_index('name').T.to_dict('list')

    return img, results_dict

def get_center_point(coordinate_dict):
    di = dict()

    for key in coordinate_dict.keys():
        xmin, ymin, xmax, ymax,conf,idx = coordinate_dict[key]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        di[key] = [x_center, y_center]

    return di

def find_miss_corner(coordinate_dict):
    ref = {'bottom_right', 'top_right', 'top_left', 'bottom_left'}
    source = set(coordinate_dict.keys())
    diff = list(ref - source)

    return diff

def calculate_missed_coord_corner(coordinate_dict):
    thresh = 0

    diff = find_miss_corner(coordinate_dict)

    if len(diff) > 1:
        print("Cannot Align Image, try to choose another one")
        exit(0)
    if len(diff) == 0:
        return coordinate_dict

    miss = diff[0]
    # calculate missed corner coordinate
    if miss == 'top_left':
        midpoint = np.add(coordinate_dict['top_right'], coordinate_dict['bottom_left']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_right'][0] - thresh
        coordinate_dict['top_left'] = [x, y]
    elif miss == 'top_right':
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_left'][0] - thresh
        coordinate_dict['top_right'] = [x, y]
    elif miss == 'bottom_right':
        midpoint = np.add(coordinate_dict['bottom_left'], coordinate_dict['top_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_left'][0] - thresh
        coordinate_dict['bottom_right'] = [x, y]
    elif miss == 'bottom_left':
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_right'][0] - thresh
        coordinate_dict['bottom_left'] = [x, y]

    return coordinate_dict

def perspective_transform(image, source_points):
    top_left = list(map(int, source_points['top_left']))
    top_right = list(map(int, source_points['top_right']))
    bottom_right = list(map(int, source_points['bottom_right']))
    bottom_left = list(map(int, source_points['bottom_left']))

    source_points = np.float32([top_left, top_right, bottom_right, bottom_left])
    dest_points = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (640,640))

    return dst

def main():
    img, results_dict = predict_results()
    di = get_center_point(results_dict)
    full_corners = calculate_missed_coord_corner(di)
    dst = perspective_transform(img, full_corners)

    cv2.imwrite("1.jpg", dst)

if __name__ == '__main__':
    main()

