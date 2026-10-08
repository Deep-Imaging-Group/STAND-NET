import os
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.init as init
from tqdm import tqdm
from loguru import logger
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import visdom

from util.tools import get_class_or_function_name, remove_file, get_initial_chars
from util.logger import Recoder
from util.tools import get_formated_args

class BaseModel(nn.Module):
    def __init__(self, args, training) -> None:
        super().__init__()
        if training:
            self._set_training_config(args)

        self._args = args
        self.device = args.device
    
    def _set_training_config(self, args):
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.metrics = []
        self._total_step = 0

        self.TIMESTAMP = "{0:%Y-%m-%d %H-%M-%S/}".format(datetime.now())
        # path = os.path.join(args.checkpoint, "log", self.TIMESTAMP)
        path = os.path.join(args.checkpoint, "log")
        self.log_path = path
        self._writer = SummaryWriter(path)
        self._writer.close()

        name = "{}-{}".format(args.model_name, args.args_hash_code)
        self._vis = visdom.Visdom(server=args.vis_server, env=name, use_incoming_socket=False, print_exception=False)




        self._recoder = Recoder()

        self._cur_epoch = 0
        self._best_val_loss = -1
    
    def fit(self, trainset:DataLoader, valset:DataLoader=None, epochs=1, print_by_step=-1):
        r"""performance the optimization phase

        Args:
            trainset: the trainset dataloader
            valset: the valset dataloader
            epochs: total training epochs
            print_by_step: how many step to display training information
        """
        self._trainset = trainset
        self._valset = valset
        
        for epoch in range(self._cur_epoch, epochs):
            self._cur_epoch = epoch
            self.train_one_epoch(epoch, print_by_step)
            val_metric_values = self.val_one_epoch(epoch)
            self.save_state_dict(val_metric_values)
            if hasattr(self._args, "lr_scheduler") and self._args.lr_scheduler:
                self.lr_scheduler.step(val_metric_values["loss"])


    def train_one_epoch(self, epoch, print_by_step):
        r"""performance the optimization phase one epoch

        Args:
            epoch: the currented epoch
            print_by_step: how many step to display training information

        """
        if self.criterion is None:
            raise Exception("You must call 'model.criterion = criterion' to set criterion")

        if self.optimizer is None:
            raise Exception("You must call 'model.optimizer = optimizer' to set optimizer")

        self.train()
        # for cur_step, data in enumerate(tqdm(self._trainset, desc="Epoch={}".format(epoch))):
        gradient_clip_log_flag = True

        for cur_step, data in enumerate(self._trainset):
            # if cur_step == 9:
            #     _x_ = 1
            # optimize the model
            input, pred, target = self.step(data, is_prediction=False, device=self.device)
            loss = self.compute_loss(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            if hasattr(self._args, "gradient_clip") and self._args.gradient_clip:
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=2)
                if epoch == 0 and gradient_clip_log_flag:
                    logger.info("Gradient clip is used...")
                    gradient_clip_log_flag = False
            self.optimizer.step()

            metric_values = self.compute_metrics(pred.detach(), target.detach())
            metric_values["loss"] = loss.cpu().item()
            self.record_metric_history(metric_values)

            # log the training information
            if cur_step % print_by_step == 0 or cur_step + 1 == len(self._trainset):
                metric_info = self.get_metric_info(metric_values)
                logger.info("{}-Train: epoch={}, step={}/{}, {}".format(self._args.args_hash_code[:8], epoch, cur_step+1, len(self._trainset), metric_info))

                # self.vis_image(input, pred, target)
            
            self.vis_image(input, pred, target)
            self._total_step += 1
        metric_mean_value = self._recoder.summary()
        # self._writer.add_scalars("train", metric_mean_value, epoch)
        self._write_metric_to_tensorboard("train", metric_mean_value, epoch)
        
        metric_info = self.get_metric_info(metric_mean_value)
        logger.info("****************************************************************")
        logger.info("Train: epoch={}, {}".format(epoch, metric_info))
        logger.info("****************************************************************")

    def val_one_epoch(self, epoch):
        r"""valid performance on val dataset

        Args:
            epoch: the currented epoch
        """ 
        self.eval()
        for data in self._valset:
            with torch.no_grad():
                _, pred, target = self.step(data, is_prediction=False, device=self.device)
            
            loss = self.compute_loss(pred, target)
            metric_values = self.compute_metrics(pred, target)
            metric_values["loss"] = loss.cpu().item()
            self.record_metric_history(metric_values)

        metric_mean_value = self._recoder.summary()
        # self._writer.add_scalars("val", metric_mean_value, epoch)
        self._write_metric_to_tensorboard("val", metric_mean_value, epoch)

        metric_info = self.get_metric_info(metric_mean_value)
        logger.info("****************************************************************")
        logger.info("Val: epoch={}, {}".format(epoch, metric_info))
        logger.info("****************************************************************")

        return metric_mean_value
    
    def _write_metric_to_tensorboard(self, prefix, metric_mean_value, epoch):
        for key, val in metric_mean_value.items():

            self._writer.add_scalar("{}-{}".format(prefix, key), val, epoch)


    def predict(self, x):
        r"""perform forward on x, and filter some unneeded the output

        Args:
            x: the input of model
        """
        self.eval()
        with torch.no_grad():
            pred = self.step(x, is_prediction=True, device=self.device)
        return pred

    def step(self, data, is_prediction, device=None):
        r"""forward on the current batch data

        Args:
            data: the data from dataloader
        """
        
        # input = Variable(data[0]).to(self._args.device)
        # target = Variable(data[1]).to(self._args.device)
        if is_prediction:
            input = self.get_input_for_model(data, device, is_prediction)
            pred = self(*input)
            return pred
        else:
            input, target = self.get_input_for_model(data, device, is_prediction)
            pred = self(*input)
            return input, pred, target
    
    def get_dummy_input(self, dataloader: DataLoader):
        for item in dataloader:
            return item
    
    def record_metric_history(self, metric_values):
        for key, value in metric_values.items():
            self._recoder.record(key, value)
        
        return metric_values

    # def vis_image(self, *input: list[torch.tensor], nrow=3):
    #     if len(input) == 3:
    #         images = []
    #         for item in input:
    #             images.append(item)
    #         images = torch.stack(images, dim=0)
    #         images = images.detach().cpu().abs()
    #         self._vis.images(images, nrow=nrow, padding=10, win=self.TIMESTAMP, opts={"title": self.TIMESTAMP})
    #     else:
    #         index = 0
    #         images = []
    #         for item in input:
    #             images.append(item[index])
    #         images = torch.stack(images, dim=0)
    #         images = images.detach().cpu().abs()
    #         self._vis.images(images, nrow=nrow, padding=10, win=self.TIMESTAMP, opts={"title": self.TIMESTAMP})
    def vis_image(self, input, pred, target):
        raise Exception("Must override the this method in subclass for visualing results.")

    def compute_metrics(self, pred, target):
        r"""compute the loos between the output and target

        Args:
            pred: the output of the model
            target: the corresponding target
        
        """
        if self.metrics is None:
            return
        metrics_values = {}

        pred_for_metric = self.get_pred_for_metric(pred) # maybe some of the output of the model is not used to compute metric
        target_for_metric = self.get_target_for_metric(target)
        for metric in self.metrics:
            key = get_class_or_function_name(metric)

            if pred_for_metric.is_complex():
                pred_for_metric = pred_for_metric.abs()
            
            if target_for_metric.is_complex():
                target_for_metric = target_for_metric.abs()


            value = metric(pred_for_metric, target_for_metric)
            if value.numel() == 1:
                value = [value.cpu().item()]
            else:
                value = value.cpu().detach().numpy().tolist()
            metrics_values[key] = value
        
        return metrics_values

    def compute_loss(self, pred, target):
        r"""compute the loos between the output and target

        Args:
            pred: the output of the model
            target: the corresponding target
        
        """
        pred_for_loss = self.get_pred_for_loss(pred)    # maybe some of the output of the model is not used to compute loss
        loss = self.criterion(pred_for_loss.abs(), target.abs())
        return loss
    
    def save_state_dict(self, cur_val_metrics):
        r"""save the state dict

        Args:
            cur_val_metrics: the metric value on val dataset, used to compare with the previous bestval metric values
        
        """
        def save(path):
            state_dicts = {
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": self._cur_epoch,
                    "best_val_loss": self._best_val_loss,
                    "final_val_loss": cur_val_metrics["loss"],
                    "best_info": self.best_info
                }
            if self.lr_scheduler is not None:
                state_dicts["le_scheduler"] = self.lr_scheduler.state_dict()
            torch.save(state_dicts, path)
        
        if self._best_val_loss == -1 or self._best_val_loss > cur_val_metrics["loss"]:
            path = os.path.join(self._args.checkpoint, "best_*.pkl")
            remove_file(path)

            self._best_val_loss = cur_val_metrics["loss"]
            file_name = "best_epoch={}".format(self._cur_epoch)
            for key, val in cur_val_metrics.items():
                metric_str = "_{}={:.4f}".format(get_initial_chars(key), val)
                file_name += metric_str
            file_name += ".pkl"

            # only used to send email
            self.best_info = file_name


            path = os.path.join(self._args.checkpoint, file_name)
            save(path)
        
        path = os.path.join(self._args.checkpoint, "final_*.pkl")
        remove_file(path)

        # self._best_val_loss = cur_val_metrics["loss"]
        file_name = "final_loss={:.4f}=_epoch={}=.pkl".format(cur_val_metrics["loss"], self._cur_epoch)
        path = os.path.join(self._args.checkpoint, file_name)
        save(path)

        file_name = "final_epoch={}".format(self._cur_epoch)
        for key, val in cur_val_metrics.items():
            metric_str = "_{}={:.4f}".format(get_initial_chars(key), val)
            file_name += metric_str
        file_name += ".pkl"
        path = os.path.join(self._args.checkpoint, file_name)
        save(path)

        # only used to send email
        self.final_info = file_name

        file_name = "epoch={}".format(self._cur_epoch)
        for key, val in cur_val_metrics.items():
            metric_str = "_{}={:.4f}".format(get_initial_chars(key), val)
            file_name += metric_str
        file_name += ".pkl"
        path = os.path.join(self._args.checkpoint, file_name)
        save(path)

    def restore_state_dict(self, path, training):
        r"""restore state_dict from checkpoint

        Args:
            path: the path of the checkpoint
        """
        state_dicts = torch.load(path, map_location=self._args.device)
        if training:
            self.load_state_dict(state_dicts["model_state_dict"])
            self.optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
            self._cur_epoch = state_dicts["epoch"] + 1
            logger.info("current epoch: {}".format(self._cur_epoch))
            if "best_val_loss" in state_dicts:
                self._best_val_loss = state_dicts["best_val_loss"]
                logger.info("best_val_loss: {}".format(self._best_val_loss))
            if "best_info" in state_dicts:
                self.best_info = state_dicts["best_info"]

            if "lr_scheduler" in state_dicts:
                self.lr_scheduler.load_state_dict(state_dicts["lr_scheduler"])
        else:
            self.load_state_dict(state_dicts["model_state_dict"])
        logger.info("Restore state dict from {}".format(path))
    
    def get_pred_for_loss(self, pred):
        """
        Usually, the model return multiple outpus for experimental convenience, e.g. extract the intermediate feature map, but only partial pred is used to compute loss.

        Args:
            pred: the output of the model
        """
        return pred
    
    
    def get_input_for_model(self, input, device, is_prediction):
        raise Exception("Must override the this method in subclass.")

    def get_pred_for_metric(self, pred):
        """
        Usually, the model return multiple outpus for experimental convenience, e.g. extract the intermediate feature map, but only partial pred is used to compute metric.

        Args:
            pred: the output of the model
        """
        return pred
    
    def get_target_for_metric(self, target):
        """
        get target for metric

        Args:
            target: the output of the model
        """
        return target


    
    def get_pred_for_vis(self, pred):
        r"""Usually, the model return multiple outpus for experimental convenience, e.g. extract the intermediate feature map, but only one pred is used to visual.

        Args:
            pred: the output of the model
        """
        return pred

    def get_metric_info(self, metric_values:dict):
        r"""convert the metric values dict to a formated string

        Args:
            metric_values: the computeed metric values, the key is the metric name, the value is the corresponding value
        
        Example:
            metric_values = {"loss": 3.333, "accuracy": 0.98}
            return "loss=0.333, accuracy=0.98"
        
        """
        info = []

        for key, value in metric_values.items():
            if isinstance(value, tuple) or isinstance(value, list):
                value = sum(value) / len(value)
            info.append("{}={:.4f}".format(key, value))

        return ", ".join(info)
    
    def add_metric(self, metric):
        if not hasattr(self, "metrics"):
            self.metrics = []
        self.metrics.append(metric)
    
    def initialize(self, dummy_input=None):
        if dummy_input is not None:
            self.step(dummy_input, device="cpu")  # This operation is used to construct grap for lazy model.

        def weight_init(m):
            '''
            Usage:
                model = Model()
                model.apply(weight_init)
            '''
            if isinstance(m, nn.Conv1d):
                init.normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.Conv3d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose1d):
                init.normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose3d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)
            elif isinstance(m, nn.LSTMCell):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)
            elif isinstance(m, nn.GRUCell):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)
        self.apply(weight_init)
        logger.info("Initialize model parameter......")
    def close(self):
        self._vis.close()
        self._writer.close()
        
        path = os.path.join(self._args.project_root_path, "checkpoints", "email.txt")
        while os.path.exists(path):
            time.sleep(2)

        logger.info("****************************************************************")
        logger.info("{}".format(self.best_info))
        logger.info("****************************************************************")
        with open(path, "w") as f:
            if hasattr(self, "best_info") and hasattr(self, "final_info"):
                f.write("{}\n".format(self.best_info))
                f.write("{}\n".format(self.final_info))
                f.write("{}\n".format(self._args.args_hash_code))
                f.write("{}\n".format(get_formated_args(self._args)))
