import torch
import torch.nn.functional as F

import torch.optim as optim
from utils.utils import AverageMeter
import os
import time

from tqdm import tqdm
from data.transforms import VisualTransform, get_augmentation_transforms
from data.hdf5_data_loader import get_dataset_from_list
from models.bc.network import build_net
from models.base import BaseTrainer
import logging
import pdb
import importlib
from models.ghostnet130_MH.network import RGBDMH
from models.ghostnet130_MH.loss_fn import CMFL
from torchvision import transforms
import torch.nn as nn
import numpy as np



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
        batch_size = 64
        num_workers = 4
        epochs=25
        learning_rate=0.0001
        weight_decay = 0.00001
        super().__init__(config)     
        self.config = config
        self.network = RGBDMH(pretrained=True,num_channels=4)
        if self.config.CUDA:
            self.network.cuda()
        for name,param in  self.network.named_parameters():
	        param.requires_grad = True
        self.cmfl_loss = CMFL(alpha=1, gamma= 3, binary= False, multiplier=2)
        self.bce_loss = nn.BCELoss()

        self.optimizer = optim.Adam(
           filter(lambda p: p.requires_grad, self.network.parameters()), lr=learning_rate,weight_decay=weight_decay
        )
    def __get_dataset___(self):
        self.Dataset = importlib.import_module('data.hdf5_dataset').__dict__[self.config.DATA.DATASET]
        return self.Dataset

    def init_dataloader(self):
        config = self.config

        batch_size = config.DATA.BATCH_SIZE
        num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS

        dataset_root_dir = config.DATA.ROOT_DIR
        rgb_path = "WMCA_preprocessed_RGB/WMCA/"
        cdit_path = "WMCA_preprocessed_CDIT/WMCA/"
        #dataset_subdir = config.DATA.SUB_DIR  # 'EXT0.2'
        dataset_RGB = os.path.join(dataset_root_dir, rgb_path)
        dataset_cdit = os.path.join(dataset_root_dir, cdit_path)
        
        test_data_transform =  transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor()])
        Dataset = self.__get_dataset___()


        if not self.train_mode:
            assert config.DATA.TEST, "Please provide at least a data_list"
            test_dataset = get_dataset_from_list(config.DATA.TEST, Dataset, test_data_transform, num_frames=config.DATA.NUM_FRAMES, rgb_dir=dataset_RGB,cdit_dir=dataset_cdit)
            self.test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
                                                                shuffle=False)

        else:
            assert config.DATA.TRAIN, "CONFIG.DATA.TRAIN should be provided"
            aug_transform = get_augmentation_transforms(config)
            train_data_transform = VisualTransform(config, aug_transform)
            #train_data_transform =  transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor()])
            # transforms already within visualTransform
            train_dataset = get_dataset_from_list(config.DATA.TRAIN, Dataset, train_data_transform, num_frames=config.DATA.NUM_FRAMES, rgb_dir=dataset_RGB,cdit_dir=dataset_cdit)
            self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size, num_workers=num_workers,shuffle=True)


            assert config.DATA.VAL, "CONFIG.DATA.VAL should be provided"
            val_dataset = get_dataset_from_list(config.DATA.VAL, Dataset, test_data_transform, num_frames=config.DATA.NUM_FRAMES, rgb_dir=dataset_RGB,cdit_dir=dataset_cdit)
            self.val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=num_workers,shuffle=False)




    def _train_one_epoch(self, epoch, train_data_loader):
        losses = AverageMeter()
        self.network.train()
        num_train = len(train_data_loader) * self.batch_size

        with tqdm(total=num_train) as pbar:
            for i, batch_data in enumerate(train_data_loader):
                self.optimizer.zero_grad()
                loss,output_prob,_,_ = self.compute_loss(self.network,batch_data)
                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                pbar.set_description(
                    (
                        " total loss={:.3f} ".format(loss.item(),
                                                     )
                    )
                )
                pbar.update(self.batch_size)

                # log to tensorboard
                if self.tensorboard:
                    self.tensorboard.add_scalar('loss/train_total', loss.item(), self.global_step)

                self.global_step += 1
            return losses.avg

    def compute_loss(self,network,img):
        """
        Compute the losses, given the network, data and labels and 
        device in which the computation will be performed. 
        """
        network_input, target = img[1], img[2]
        target = target.cuda()

        beta = 0.5

        op, op_rgb, op_d = network(network_input.cuda())

        loss_cmfl = self.cmfl_loss(op_rgb,op_d,target.unsqueeze(1).float())

        loss_bce = self.bce_loss(op,target.unsqueeze(1).float())

        loss= beta*loss_cmfl +(1-beta)*loss_bce

  

        return loss,op,op_rgb,op_d

    def test_inference(self,network,img,starter,ender):
        """
        Compute the losses, given the network, data and labels and 
        device in which the computation will be performed. 
        """
        network_input, target = img[1], img[2]
        target = target.cuda()

        beta = 0.5

        # import time
        # tic = time.time()
        starter.record()
        op, op_rgb, op_d = network(network_input.cuda())
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        # toc = time.time()
        # inference_time = toc - tic; # if batch size is 32, N batches; store somewhere and average over 1 epoch (only during test)
        loss_cmfl = self.cmfl_loss(op_rgb,op_d,target.unsqueeze(1).float())

        loss_bce = self.bce_loss(op,target.unsqueeze(1).float())

        loss= beta*loss_cmfl +(1-beta)*loss_bce

  

        return loss,op,op_rgb,op_d,curr_time/24
    
    def test(self, test_data_loader):
        avg_test_loss = AverageMeter()
        scores_pred_dict = {}
        face_label_gt_dict = {}
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings=np.zeros((len(test_data_loader),1))
        timer_index = 0     

        self.network.eval()
        with torch.no_grad():
            for data in tqdm(test_data_loader):
                frameindex,network_input,target =  data[0],data[1],data[2]

                test_loss,output_prob,op_rgb,op_d,inference_time = self.test_inference(self.network,data,starter,ender
                                             )
                timings[timer_index] = inference_time
                timer_index+=1
                #select which scores to use , and change pred_score passed into collect_scores_from_loader
                pred_score,pred_score_rgb,pred_score_d = self._get_score_from_prob(output_prob,op_rgb,op_d)
                avg_test_loss.update(test_loss.item(), network_input.size()[0])

                gt_dict, pred_dict = self._collect_scores_from_loader(scores_pred_dict, face_label_gt_dict,
                                                                      data[2].numpy(), pred_score_d,
                                                                      frameindex
                                                                      )
        mean_syn = np.sum(timings) / len(test_data_loader)
        std_syn = np.std(timings)
        logging.info("Mean inference time :{} ".format(mean_syn))
        logging.info("Std of inference time :{} ".format(std_syn))
                
        test_results = {
            'scores_gt': gt_dict,
            'scores_pred': pred_dict,
            'avg_loss': avg_test_loss.avg,
            'mean_inf_time':mean_syn,
            'std_inf_time':std_syn,
        }
        return test_results
    def _get_score_from_prob(self, output_prob,op_rgb,op_d):
        #output_scores = torch.softmax(output_prob, 1)
        output_scores = output_prob.cpu().numpy()
        output_scores_rgb = op_rgb.cpu().numpy()
        output_scores_d = op_d.cpu().numpy()

        return output_scores,output_scores_rgb,output_scores_d










