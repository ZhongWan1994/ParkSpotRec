import copy
import os.path
from configparser import ConfigParser

import torch
import torch.nn.functional as F
import torch.nn.functional as loss_func
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from utils import Utils


class NetTrainer(object):
    def __init__(self, conf: ConfigParser, dataloader: DataLoader, phase="train"):
        self.conf = conf
        self.dataloader = dataloader
        self.phase = phase
        if self.phase == "train":
            self.use_pretrained = bool(self.conf.get("model", "use_pretrained"))
            self.feature_extracting = bool(self.conf.get("model", "feature_extract"))
            self.lr = float(self.conf.get("model", "lr"))
            self.early_stop_tolerance = int(self.conf.get("model", "early_stop_tolerance"))
            self.model = self.select_model()
            self.set_parameter_requires_grad()
            self.model = self.set_last_layer(self.model)
            self.param_to_train = self.get_param_to_self_train()
            self.optim = optim.Adam(self.param_to_train, lr=self.lr)
            self.schd = optim.lr_scheduler.StepLR(self.optim, step_size=30, gamma=0.1)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def select_model(self):
        model_name = self.conf.get("model", "model_name")
        checkpoint_dir = self.conf.get("output", "checkpoint")
        if os.path.exists(checkpoint_dir):
            model = models.resnet50()
            print("%s model weight file exists, load it..." % model_name)
            model = Utils.load_checkpoint(checkpoint_dir, model, model_name)
            return model
        os.environ['TORCH_HOME'] = '/home/wz/PycharmProjects/park_repetition/checkpoint'
        model = models.resnet50(pretrained=self.use_pretrained)
        return model

    def set_last_layer(self, model):
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        for param in model.fc.parameters():
            param.requires_grad = True
        if self.phase =="train":
            print("params in fc_layer will be trained")
        return model

    def set_parameter_requires_grad(self):
        if self.feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    def get_param_to_self_train(self):
        params_to_update = self.model.parameters()
        if self.feature_extracting:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
            print("all params will not be trained")
        print("all params will be trained")

        return params_to_update

    @staticmethod
    def accuracy_score(true, pred):
        l = torch.eq(true, pred)
        return torch.sum(l) / len(pred)

    def save_best_model(self):
        path = self.conf.get("output", "checkpoint")
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.model.state_dict(), os.path.join(path, "best.pt"))

    def cal_loss_accuracy_for_single_batch(self, data, label):
        pred = self.model(data)
        pred_label = torch.argmax(pred, dim=1)
        label = label.view(pred.shape[0], )
        loss = loss_func.cross_entropy(pred, label, reduction="sum")
        acc = self.accuracy_score(pred_label.cpu(), label.cpu())
        return loss, acc, {"pred_label": pred_label, "label": label}

    def train_model(self):
        epochs = int(self.conf.get("model", "epochs"))
        best_val_acc = 0
        early_stop_cnt = 0
        for epoch in range(epochs):
            print("[Epoch:%s]" % epoch)
            sum_train_loss, sum_train_acc, sum_val_loss, sum_val_acc = 0, 0, 0, 0
            train_scale, val_scale = 0, 0
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                for data, label in self.dataloader:
                    self.model.zero_grad()
                    loss, acc, _ = self.cal_loss_accuracy_for_single_batch(data, label)
                    if phase == "train":
                        loss.backward()
                        self.optim.step()
                        sum_train_acc += acc.cpu().item() * len(label)
                        sum_train_loss += loss.item()
                        train_scale += len(label)
                    if phase == "valid":
                        val_loss, val_acc, _ = self.cal_loss_accuracy_for_single_batch(data, label)
                        sum_val_acc += val_acc.cpu().item() * len(label)
                        sum_val_loss += val_loss.item()
                        val_scale += len(label)
                if phase == "train":
                    self.schd.step()

            train_loss = sum_train_loss / train_scale
            val_loss = sum_val_loss / val_scale
            avg_train_acc = sum_train_acc / train_scale
            avg_val_acc = sum_val_acc / val_scale
            print("Epoch: %d  train_loss: %.4f, train_val: %.4f, val_loss: %.4f, val_acc: %.4f" % (
                epoch, train_loss, avg_train_acc, val_loss, avg_val_acc
            ))

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                self.save_best_model()
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stop_cnt > self.early_stop_tolerance:
                break

    def test_on_valid_set(self, valid_dataloader: DataLoader):
        model = models.resnet50()
        model = self.set_last_layer(model)
        path = os.path.join(self.conf.get("output", "checkpoint"), "best.pt")
        weight = torch.load(path, weights_only=False)
        model.load_state_dict(weight)
        true_num, size = 0, 0
        for data, label in valid_dataloader:
            pred = model(data)
            softmax_res = F.softmax(pred, dim=1)
            pred_label = torch.argmax(softmax_res, dim=1)
            true_num += sum(torch.eq(pred_label, label))
            size += len(pred)
        acc = true_num / size
        print("acc on valid is:%.2f" % acc)

    def predict(self, dataloader):
        model = models.resnet50()
        model = self.set_last_layer(model)
        path = os.path.join(self.conf.get("output", "checkpoint"), "best.pt")
        weight = torch.load(path, weights_only=False)
        model.load_state_dict(weight)

        pred_list = []
        for data in dataloader:
            pred = model(data)
            softmax_res = F.softmax(pred, dim=1)
            pred_label = torch.argmax(softmax_res, dim=1)
            pred_label = [i.item() for i in pred_label]
            pred_list.extend(pred_label)
        return pred_list
