import argparse 
import os 
import sys 
from datetime import datetime

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU 
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

import albumentations as A

from tqdm import tqdm

# own codebase 
sys.path.append('/home/gentleprotector/ubs_ws24/UBS-DFC25/codebase')
import unetbase
from unetbase import UNet
import data_loader
from utils import ce_per_class


class Trainer(object): 
    def __init__(self, args): 
        # save input arguments
        self.args = args 

        # setup directory for saving model checkpoints 
        now_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        self.save_path = os.path.join(args.save_dir, args.run_name + '_' + now_str)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # setup tensorboard writer
        self.writer = SummaryWriter(
            log_dir =  os.path.join(self.save_path + '/tb_logs/')
        )

        # setup device     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # setup model from custom script and send to device
        self.model = UNet(in_channels=4, n_classes=4).float()
        self.model = self.model.to(self.device) 

        # check and load model checkpoint if resume == True 
        if args.resume is not None: 
            # check for checkpoint 
            if not os.path.isfile(args.resume):
                raise RunTimeError(
                    f'No checkpoint found at {args.resume}'
                )
            checkpoint = torch.load(args.resume)
            # change model weights to checkpoint state
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        #self.norm_mean = torch.tensor(args.norm_mean).float().to(self.device)
        #self.norm_std = torch.tensor(args.norm_std).float().to(self.device)
        # transformations 
        self.train_transforms = A.Compose([
            A.Normalize(mean=[88.26, 83.83, 75.40, 57.39], std=[38.06, 31.84, 28.87, 24.43]),
            A.RandomCrop(width=self.args.crop_size, height=self.args.crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], additional_targets={'image': 'image', 'mask': 'mask', 'coords': 'mask'}, is_check_shapes=False)

        self.val_transforms = A.Compose([
           A.Normalize(mean=[88.26, 83.83, 75.40, 57.39], std=[38.06, 31.84, 28.87, 24.43]),
        ], additional_targets={'image': 'image', 'mask': 'mask', 'coords': 'mask'}, is_check_shapes=False)


        self.weights = torch.tensor(args.class_weights).float().to(self.device)
        self.label_smoothing = torch.tensor(args.label_smoothing).float().to(self.device)
        # setup loss functions
        self.criterion_cl = nn.CrossEntropyLoss(
            weight=self.weights,
            reduction='none',
            label_smoothing=self.label_smoothing
            )

        # self.criterion_cl = smp.losses.FocalLoss(
        #     mode='multiclass',
        #     alpha=None,
        #     gamma=2.0,
        #     reduction='none'
        # )
        self.criterion_segm = smp.losses.LovaszLoss(
            mode='multiclass',
        )

        
        # setup logging variables for saving model and optimizer states
        # here start iter = 1 and loss = 0.0 (to be updated after first iter)
        self.iter =  1
        self.loss = 0.0

        # setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        #self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=16717)

        # if resume is checkpoint is available and set optimizer state accordingly
        if args.resume is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.iter = checkpoint['iter'] 
            self.total_loss = checkpoint['loss']

        # metrics 
        # too much here <- could manually calulate the mean from the per clas ious! 
        self.train_iou_classes = MeanIoU(
            num_classes=4,
            per_class = True,  
            input_format = 'index' 
            ).to(self.device)

        self.train_iou_all = MeanIoU(
            num_classes=4,
            per_class = False, 
            input_format = 'index'
            ).to(self.device)
        
        self.val_iou_classes = MeanIoU(
            num_classes=4,
            per_class = True, 
            input_format = 'index'
            ).to(self.device)
        
        self.val_iou_all = MeanIoU(
            num_classes=4,
            per_class = False, 
            input_format = 'index'
            ).to(self.device)


    def training(self): 

        print(
            f'-------------STARTING TRAINING-------------'
        )

        best_iou = 0.0 
        best_iter = 0

        train_dataset = data_loader.DFC25_TRAIN(
            dataset_path=self.args.train_data_path,
            dataset_file=self.args.train_file,
            max_iters=self.args.max_iters,
            loader=data_loader.img_loader,
            transforms=self.train_transforms,
            crop_size=self.args.crop_size,
            suffix='.tif'
        )

        train_loader = DataLoader(
            train_dataset,
            self.args.batch_size,
            shuffle=True,
        )

        pbar_train = tqdm(
            train_loader,
            desc=f'TRAIN',
            dynamic_ncols=True
        )

        mean_cl_loss = 0.0 
        segm_loss = 0.0 
        total_loss = 0.0
        loss_cl_per_class = [] 

        # num_iters already given in the dataset 
        # len trainloader = num_iters 
        for batch_idx, (img_stacks, targets, _) in enumerate(pbar_train):
            
            self.model.train() 

            # move data to device 
            inputs = img_stacks.to(self.device)
            targets = targets.long().to(self.device)
            #coords = coords.to(self.device)


            # forward pass
            outputs = self.model(inputs)

            # zero gradients
            self.optimizer.zero_grad()

            # calculate losses 
            losses_cl = self.criterion_cl(outputs, targets)
            loss_cl_per_class_batch = ce_per_class(losses_cl, targets, 4)
            mean_cl_loss = losses_cl.mean()

            segm_loss = self.criterion_segm(outputs, targets)
            total_loss = mean_cl_loss + 0.75*segm_loss
            self.loss = total_loss.item()

            # backpropagation + optimizer step
            total_loss.backward()
            self.optimizer.step()

            # metrics update and logging
            self.train_iou_classes.update(torch.argmax(outputs, dim=1), targets)
            self.train_iou_all.update(torch.argmax(outputs, dim=1), targets)
            
            
            loss_cl_per_class.append(loss_cl_per_class_batch)
            mean_cl_loss += mean_cl_loss
            segm_loss += segm_loss
            total_loss += total_loss.item()

            #class_ious = self.train_iou_classes.compute()
            #total_iou = self.train_iou_all.compute().item()
            

            # update progress bar 
            if (self.iter % 558) == 0: 

                total_iou = self.train_iou_all.compute().item()
                class_ious = self.train_iou_classes.compute()
                total_loss = total_loss / 558
                mean_cl_loss = mean_cl_loss / 558
                segm_loss = segm_loss / 558

                loss_cl_per_class = np.array(loss_cl_per_class).mean(axis=0)

                print(
                    f'\n--------------------------------------------'
                    f'\nTRAIN METRICS AFTER {self.iter} ITERATIONS'
                    f'\n--------------------------------------------'
                    f'\nTotal Loss: {total_loss.item():.4f}'
                    f'\nLoss CL: {mean_cl_loss.item():.4f}'
                    f'\nLoss Segm: {segm_loss.item():.4f}'
                    f'\nLoss Background: {loss_cl_per_class[0].item():.4f}'
                    f'\nLoss Intact: {loss_cl_per_class[1].item():.4f}'
                    f'\nLoss Damaged: {loss_cl_per_class[2].item():.4f}'
                    f'\nLoss Destroyed: {loss_cl_per_class[3].item():.4f}'
                    f'\n--------------------------------------------'
                    f'\nmIoU: {total_iou:.4f}'
                    f'\nIoU Background: {class_ious[0].item():.4f}'
                    f'\nIoU Intact: {class_ious[1].item():.4f}'
                    f'\nIoU Damaged: {class_ious[2].item():.4f}'
                    f'\nIoU Destroyed: {class_ious[3].item():.4f}'
                    f'\n--------------------------------------------'
                )

                self.writer.add_scalar('train/loss_background', loss_cl_per_class[0].item(), self.iter)
                self.writer.add_scalar('train/loss_intact', loss_cl_per_class[1].item(), self.iter)
                self.writer.add_scalar('train/loss_damaged', loss_cl_per_class[2].item(), self.iter)
                self.writer.add_scalar('train/loss_destroyed', loss_cl_per_class[3].item(), self.iter)
                self.writer.add_scalar('train/mean_cl_loss', mean_cl_loss.item(), self.iter)
                self.writer.add_scalar('train/segm_loss', segm_loss.item(), self.iter)
                self.writer.add_scalar('train/total_loss', total_loss.item(), self.iter)
                self.writer.add_scalar('train/total_iou', total_iou, self.iter)
                self.writer.add_scalar('train/iou_background', class_ious[0].item(), self.iter)
                self.writer.add_scalar('train/iou_intact', class_ious[1].item(), self.iter)
                self.writer.add_scalar('train/iou_damaged', class_ious[2].item(), self.iter)
                self.writer.add_scalar('train/iou_destroyed', class_ious[3].item(), self.iter)

                mean_cl_loss = 0.0
                segm_loss = 0.0
                total_loss = 0.0
                loss_cl_per_class = []

                # reset metrics for next 10 iterations
                self.train_iou_classes.reset()
                self.train_iou_all.reset()

                if (self.iter % 1116) == 0: 

                    self.model.eval()

                    val_miou, bg_iou, intact_iou, dmg_iou, destr_iou = self.validation() 

                    if val_miou > best_iou:
                        torch.save({
                            'iteration': self.iter,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': self.loss
                        }, os.path.join(self.save_path, 'best_model.pth'))


                        best_iou = val_miou
                        best_iter = self.iter
                        best_round = {
                            'Best iteration': best_iter,
                            'Best mIoU': best_iou,
                            'Background IoU': bg_iou,
                            'Intact IoU': intact_iou,
                            'Damaged IoU': dmg_iou,
                            'Destroyed IoU': destr_iou
                        }

                        print(
                            f'\n--------------------------------------------'
                            f'\nBEST MODEL AFTER {self.iter} ITERATIONS'
                            f'\n--------------------------------------------'
                            f'\nBest mIoU: {best_iou:.4f}'
                            f'\nBackground IoU: {bg_iou:.4f}'
                            f'\nIntact IoU: {intact_iou:.4f}'
                            f'\nDamaged IoU: {dmg_iou:.4f}'
                            f'\nDestroyed IoU: {destr_iou:.4f}'
                        )

            # update iteration and total loss 
            self.iter +=1
            #self.scheduler.step()
            


    def validation(self):
        print(
            f'-------------VALIDATION PHASE-------------'
        )


        val_dataset = data_loader.DFC25_VAL(
            dataset_path=self.args.val_data_path,
            dataset_file=self.args.val_file,
            loader=data_loader.img_loader,
            max_iters=None,
            transforms=self.val_transforms,
            suffix='.tif'
        )

        val_loader = DataLoader(
            val_dataset,
            self.args.batch_size // 2,
            shuffle=False,
        )

        torch.cuda.empty_cache()

        # set running variables for the losses to log 
        val_total_loss = 0.0
        val_cl_loss = 0.0
        val_segm_loss = 0.0
        val_background_loss = 0.0
        val_intact_loss = 0.0
        val_damaged_loss = 0.0
        val_destroyed_loss = 0.0

        pbar_val = tqdm(val_loader, desc=f'VAL', dynamic_ncols=True)

        with torch.no_grad():
            # here all images usd (no max_iters)
            for batch_idx, (img_stacks, targets, _) in enumerate(pbar_val):
                # move data to device 
                inputs = img_stacks.to(self.device)
                targets = targets.long().to(self.device)
                #coords = coords.to(self.device)

                # forward pass
                outputs = self.model(inputs)

                # update step 
                # calculate losses 
                losses_cl = self.criterion_cl(outputs, targets)
                loss_cl_per_class = ce_per_class(losses_cl, targets, 4)

                mean_cl_loss = losses_cl.mean()

                segm_loss = self.criterion_segm(outputs, targets)
                total_loss = mean_cl_loss + 0.75*segm_loss

                # metrics update and logging
                self.val_iou_classes.update(torch.argmax(outputs, dim=1), targets)
                self.val_iou_all.update(torch.argmax(outputs, dim=1), targets)

                val_total_loss += total_loss.item() 
                val_cl_loss += mean_cl_loss.item()
                val_segm_loss += segm_loss.item()
                val_background_loss += loss_cl_per_class[0]
                val_intact_loss += loss_cl_per_class[1]
                val_damaged_loss += loss_cl_per_class[2]
                val_destroyed_loss += loss_cl_per_class[3]

        
        val_total_loss = val_total_loss / len(val_loader)
        val_cl_loss = val_cl_loss / len(val_loader)
        val_segm_loss = val_segm_loss / len(val_loader)
        val_background_loss = val_background_loss / len(val_loader)
        val_intact_loss = val_intact_loss / len(val_loader)
        val_damaged_loss = val_damaged_loss / len(val_loader)
        val_destroyed_loss = val_destroyed_loss / len(val_loader)

        val_iou_classes = self.val_iou_classes.compute()
        val_iou_all = self.val_iou_all.compute()

        print(
            f'\n--------------------------------------------'
            f'\nVAL METRICS AFTER {self.iter} ITERATIONS'
            f'\n--------------------------------------------'
            f'\nTotal Loss: {val_total_loss:.4f}'
            f'\nLoss CL: {val_cl_loss:.4f}'
            f'\nLoss Segm: {val_segm_loss:.4f}'
            f'\nLoss Background: {val_background_loss:.4f}'
            f'\nLoss Intact: {val_intact_loss:.4f}'	
            f'\nLoss Damaged: {val_damaged_loss:.4f}'
            f'\nLoss Destroyed: {val_destroyed_loss:.4f}'
            f'\n--------------------------------------------'
            f'\nmIoU: {val_iou_all:.4f}'
            f'\nIoU Background: {val_iou_classes[0].item():.4f}'
            f'\nIoU Intact: {val_iou_classes[1].item():.4f}'
            f'\nIoU Damaged: {val_iou_classes[2].item():.4f}'
            f'\nIoU Destroyed: {val_iou_classes[3].item():.4f}'
            f'\n--------------------------------------------'
        )

        # log to tensorboard
        # self.writer.add_scalar('val/loss_background', val_background_loss, self.iter)
        # self.writer.add_scalar('val/loss_intact', val_intact_loss, self.iter)
        # self.writer.add_scalar('val/loss_damaged', val_damaged_loss, self.iter)
        # self.writer.add_scalar('val/loss_destroyed', val_destroyed_loss, self.iter)
        # self.writer.add_scalar('val/mean_cl_loss', val_cl_loss, self.iter)
        # self.writer.add_scalar('val/segm_loss', val_segm_loss, self.iter)
        # self.writer.add_scalar('val/total_loss', val_total_loss, self.iter)
        self.writer.add_scalar('val/total_iou', val_iou_all, self.iter)
        self.writer.add_scalar('val/iou_background', val_iou_classes[0].item(), self.iter)
        self.writer.add_scalar('val/iou_intact', val_iou_classes[1].item(), self.iter)
        self.writer.add_scalar('val/iou_damaged', val_iou_classes[2].item(), self.iter)
        self.writer.add_scalar('val/iou_destroyed', val_iou_classes[3].item(), self.iter)

         # reset metrics
        self.val_iou_classes.reset()
        self.val_iou_all.reset()

        print(
            f'-------------VALIDATION PHASE END-------------'
        )

        return val_iou_all, val_iou_classes[0], val_iou_classes[1].item(), val_iou_classes[2].item(), val_iou_classes[3].item()  
        

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm_mean', type=float, nargs='+', help='mean values for normalization', required=True)
    parser.add_argument('--norm_std', type=float, nargs='+', help='std values for normalization', required=True)
    parser.add_argument('--crop_size', type=int, default=256, help='crop size')


    parser.add_argument('--train_data_path', type=str, help='path to training data', required=True)
    parser.add_argument('--train_file', type=str, help='file containing training data list', required=True)
    parser.add_argument('--val_data_path', type=str, help='path to validation data', required=True)
    parser.add_argument('--val_file', type=str, help='file containing validation data list', required=True)

    parser.add_argument('--max_iters', type=int, help='maximum number of iterations', required=False)
    parser.add_argument('--batch_size', type=int, default=4, help='batch size', required=True)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate', required=True)
    parser.add_argument('--class_weights', type=float, nargs='+', help='class weights', required=False)
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing', required=False)

    parser.add_argument('--save_dir', type=str, help='directory to save model checkpoints', default='DFC25LOGS', required=True)
    parser.add_argument('--run_name', type=str, help='name of run', default='DFC25RUN', required=True)
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume training', default=None, required=False)

    args = parser.parse_args()

    trainer = Trainer(args)

    trainer.training() 