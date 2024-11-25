import configparser
from torch.utils.data import DataLoader
from data_lord import DataLord
from net_trainer import NetTrainer

if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read("config.ini")
    phase = conf.get("model", "phase")

    train_data_dir = conf.get("data", "train_data_dir")
    pk_train_data_ds = DataLord(train_data_dir)
    data = pk_train_data_ds.train_data
    pk_data_dl = DataLoader(data, 64, True)
    if phase == "train":
        net_trainer = NetTrainer(conf, pk_data_dl, phase)
        net_trainer.train_model()
    if phase == "test":
        # test on valid set
        valid_data = pk_train_data_ds.valid_data
        pk_valid_data_dl = DataLoader(valid_data, 64, True)
        net_trainer = NetTrainer(conf, pk_data_dl, phase)
        net_trainer.test_on_valid_set(pk_data_dl)
