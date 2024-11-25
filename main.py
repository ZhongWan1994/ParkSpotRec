



import configparser

from park import ParkImgProcessor

if __name__ == '__main__':
    processor = ParkImgProcessor()
    config = configparser.ConfigParser()
    config.read("config.ini")
    video_name = config.get("data", "video")

    class_dictionary = {"0": "empty", "1": "occupied"}
    processor.predict_on_video('/home/wz/PycharmProjects/park/parking_video.mp4', class_dictionary, config)