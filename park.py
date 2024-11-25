import configparser

import cv2
import numpy as np
from numpy.ma.core import make_mask
from sympy.plotting.intervalmath import interval
from torch.utils.data import DataLoader
from triton.profiler import start

from data_lord import DataLord
from net_trainer import NetTrainer
from utils import Utils

SPOT_AREA = [
    [[96, 484], [129,631]],
    [[167,469], [197, 627]],
    [[197,451], [230,627]],
    [[263,427], [294,626]],
    [[294,427], [321,626]],
    [[356,392], [384,629]],
    [[384,392], [410,629]],
    [[458,348], [483,628]],
    [[487,316], [509,628]],
    [[552,259], [580,636]],
    [[580,229], [605,636]],
    [[644,165], [674,633]],
    [[674,134], [699,633]],
    [[738,75], [764,632]],
    [[764,75], [791,632]],
    [[831,75], [859,632]],
    [[859,75], [886,632]],
    [[922,88], [952,632]],
    [[952,88], [980,632]],
    [[1015,161], [1045,639]],
    [[1045,161], [1073,639]],
    [[1110,117], [1141,685]],
]
SPOT_NUM = [9,10,11,13,13,19,19,18,20,24,26,30,32,36,36,36,36,35,35,31,31,37]

class ParkImgProcessor(object):
    def __init__(self):
        pass

    @staticmethod
    def trans2gray(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def canny(img):
        img = cv2.Canny(img, 127, 255)
        return img

    @staticmethod
    def poly_mask(img):
        mask = np.zeros_like(img)
        vertices = np.array([[91, 480], [484, 307],[734, 70],[990, 68],
                             [990, 147], [1093, 147], [1093, 114], [1149,109],
                             [1159, 683], [1093, 680], [1090, 649], [77, 651]], np.int32)
        cv2.fillPoly(mask, [vertices], 255)
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        return masked_image
    @staticmethod
    def crop_rectangle(img, rect_vertices):
        pts = np.array(rect_vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))
        x_vals = [pt[0][0] for pt in pts]
        y_vals = [pt[0][1] for pt in pts]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        cropped_image = img[y_min:y_max, x_min:x_max]
        return cropped_image

    @staticmethod
    def split_area(img):
        areas = []
        for i, rec_tp in enumerate(SPOT_AREA):
            spot_num = SPOT_NUM[i]
            p1, p3 = rec_tp
            w = abs(p1[0] - p3[0])
            h = abs(p1[1] - p3[1])

            p2 = [p1[0] + w, p1[1]]
            p4 = [p3[0] - w, p3[1]]

            # rec = np.array([p1,p2,p3,p4])
            # cv2.polylines(img, rec.reshape((-1, 1, 2)), True, (0, 255, 0), 2,3)
            # cv2.line(img, p1, p2, (0, 255, 0), 1)

            p1_b, p2_b = p1.copy(), p2.copy()
            p1_a, p2_a = p1_b.copy(), p2_b.copy()
            for j in range(spot_num):
                p1_b[1] = int(h / spot_num * (j + 1) + p1[1])
                p2_b[1] = int(h / spot_num * (j + 1) + p2[1])

                areas.append([p1_a, p2_a, p2_b.copy(), p1_b.copy()])
                p1_a, p2_a = p1_b.copy(), p2_b.copy()
                # cv2.line(img, p1_b, p2_b, (0, 255, 0),1)
        return areas

    def pred_single_img(self, image, conf):
        spot_areas = self.split_area(image)
        small_area_images = [self.crop_rectangle(image, spot_area) for spot_area in spot_areas]
        dl = DataLord()
        test_data = dl.input_images(small_area_images)
        pk_data_dl = DataLoader(test_data, 64, False)
        net_trainer = NetTrainer(conf, pk_data_dl, "test")
        pred = net_trainer.predict(pk_data_dl)
        return pred, small_area_images

    def predict_on_video(self, video_name, conf, ret=True):
        cap = cv2.VideoCapture(video_name)
        count = 0
        while ret:
            ret, image = cap.read()
            count += 1
            if count == 5:
                count = 0
                new_image = np.copy(image)
                overlay = np.copy(image)
                cnt_empty = 0
                color = [0, 255, 0]
                alpha = 0.5

                my_pred, small = self.pred_single_img(new_image, conf)
                all_spots = len(small)
                areas = self.split_area(new_image)
                for _pred, area in zip(my_pred, areas):
                    if _pred == 0:
                        cv2.rectangle(new_image, area[0], area[2], color, -1)
                        cnt_empty += 1

                cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

                cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

                cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.imshow('Video Capture', new_image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()

if __name__ == '__main__':
    conf = configparser.ConfigParser()
    processor = ParkImgProcessor()
    conf.read("config.ini")
    video_name = conf.get("data", "video")
    class_dictionary = {"0": "empty", "1": "occupied"}
    processor.predict_on_video('/home/wz/PycharmProjects/park/parking_video.mp4', class_dictionary, conf)
