import cv2
import numpy as np
import torch
import craft_utils
import imgproc
import craft 

class CraftTextDetector:
    def __init__(self, craft_model_path, link_refinement=False, cuda=True):
        self.craft_model = craft.CRAFT()
        self.link_refinement = link_refinement
        if cuda and torch.cuda.is_available():
            self.craft_model.load_state_dict(
                craft_utils.copyStateDict(torch.load(craft_model_path)))
            self.craft_model = self.craft_model.cuda()
        else:
            self.craft_model.load_state_dict(
                craft_utils.copyStateDict(torch.load(craft_model_path, map_location='cpu')))
            self.craft_model = self.craft_model.cpu()
        self.craft_model.eval()

    def detect_text(self, image_path):
        image = cv2.imread(image_path)
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 
            1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio

        with torch.no_grad():
            image_tensor = imgproc.image_to_tensor(img_resized)
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            y, _ = self.craft_model(image_tensor)
            y = craft_utils.uncrop(y.cpu(), size_heatmap, img_resized.shape)
            boxes, polys = craft_utils.getDetBoxes(y, self.craft_model.net, threshold=0.7)
            boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
            polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
            if self.link_refinement:
                boxes, polys = craft_utils.refine_boxes(boxes, polys, image_tensor)
        return boxes, polys
