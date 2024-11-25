import configparser

from park import ParkImgProcessor

if __name__ == '__main__':
    conf = configparser.ConfigParser()
    processor = ParkImgProcessor()
    conf.read("config.ini")
    video_name = conf.get("data", "video")
    class_dictionary = {"0": "empty", "1": "occupied"}
    processor.predict_on_video('parking_video.mp4', class_dictionary, conf)