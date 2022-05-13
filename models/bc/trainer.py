import torch
import torch.nn.functional as F

import torch.optim as optim
from utils.utils import AverageMeter
import os
import time

from tqdm import tqdm
from models.bc.network import build_net
from models.base import BaseTrainer
import logging
import pdb


class Trainer(BaseTrainer):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config):
        """
        Construct a new Trainer instance.
        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        super(Trainer, self).__init__(config)
        self.config = config

        self.network = build_net(self.config)
        if self.config.CUDA:
            self.network.cuda()
        self.loss = torch.nn.CrossEntropyLoss().cuda()

        #self.optimizer = optim.Adam(
        #    filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.init_lr,
        #)
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.network.parameters()),
            lr=self.init_lr,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.TRAIN.EPOCHS)











