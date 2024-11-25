import os.path
import numpy as np
from torchvision import transforms
from utils import Utils


class DataLord(object):
    def __init__(self, data_dir=None):
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if data_dir is not None:
            self.data_dir = data_dir
            self.train_data = self.load_dataset("train")
            self.valid_data = self.load_dataset("valid")


    def load_dataset(self, set_type):
        my_dir = str(os.path.join(self.data_dir, set_type))
        data = []
        for status in ["occupied", "empty"]:
            target_dir = os.path.join(my_dir, status)
            batch = Utils.read_jpg_files(target_dir)
            label = 1 if status == "occupied" else 0
            o = [[np.array(self.transform(x), dtype=np.float32), label] for x in batch]
            data.extend(o)
        return data

    def input_images(self, images):
        data = []
        images = [Utils.np_array_to_pil_image(image) for image in images]
        o = [np.array(self.transform(image), dtype=np.float32) for image in images]
        data.extend(o)
        return data
