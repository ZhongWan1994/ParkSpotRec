import glob
import os

import cv2
import numpy as np
import torch.nn
from PIL import Image


class Utils(object):
    @staticmethod
    def read_jpg_files(path, method="PIL"):
        jpg_files = []
        for file in os.listdir(path):
            if file.endswith(".jpg"):
                jpg_files.append(os.path.join(path, file))

        image_arrays = []
        for jpg_file in jpg_files:
            if method == "PIL":
                image = Image.open(jpg_file)
            else:
                image = cv2.imread(jpg_file)
            image_arrays.append(image)
        return image_arrays

    @staticmethod
    def load_checkpoint(path, model, prefix):
        path = os.path.join(path, "hub/checkpoints/%s*.pth" % prefix)
        w_file = glob.glob(path)[0]
        pretrained_weights = torch.load(w_file, weights_only=False)
        model.load_state_dict(pretrained_weights)
        return model

    @staticmethod
    def cv_show(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def np_array_to_pil_image(np_array_data):
        if np_array_data.dtype != np.uint8:
            if len(np_array_data.shape) == 2:  # 单通道灰度图
                np_array_data = (np_array_data / 256).astype(np.uint8)
            elif len(np_array_data.shape) == 3:  # 三通道彩色图
                np_array_data = (np_array_data / 256).astype(np.uint8)
        pil_image = Image.fromarray(np_array_data)
        return pil_image