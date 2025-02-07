import argparse 
import os 
import sys 
from datetime import datetime

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU 
import segmentation_models_pytorch as smp

import albumentations as A
from tqdm import tqdm

import torch.autograd.profiler as profiler

from torch.utils.tensorboard import SummaryWriter

# own codebase 
sys.path.append('/home/gentleprotector/ubs_ws24/UBS-DFC25/codebase')
import unetbase
from unetbase import UNet


import imageio.v2 as imageio
import rasterio as rio
from torch.utils.data import Dataset, DataLoader



def img_loader(img_path):
    img = np.array(imageio.imread(img_path), np.float32)
    return img

class DFC25_TRAIN(Dataset):
    def __init__(
        self, 
        dataset_path, 
        dataset_file,
        loader,
        transforms=None,
        suffix='.tif'
    ):
    
        self.dataset_path = dataset_path
        with open(dataset_file, 'r') as f:
            self.dataset_list = [name.strip() for name in f]

        self.loader = loader
        self.transforms = transforms
        self.suffix = suffix
    
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        pre = self.loader(os.path.join(self.dataset_path, 'pre-event',  self.dataset_list[idx] + '_pre_disaster' + self.suffix))
        post = self.loader(os.path.join(self.dataset_path, 'post-event', self.dataset_list[idx] + '_post_disaster' + self.suffix)) 
        post = np.expand_dims(post, axis=2)
        img = np.concatenate([pre, post], axis=2) 
        
        mask = self.loader(os.path.join(self.dataset_path, 'target', self.dataset_list[idx] + '_building_damage' + self.suffix))

        if self.transforms: 
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img.transpose(2, 0, 1), mask


class DFC25_VAL(Dataset):
    def __init__(
        self, 
        dataset_path, 
        dataset_file,
        loader,
        transforms=None,
        suffix='.tif'
    ):
        self.dataset_path = dataset_path
        with open(dataset_file, 'r') as f:
            self.dataset_list = [name.strip() for name in f]
        
        self.loader = loader
        self.transforms = transforms
        self.suffix = suffix
  
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        pre = self.loader(os.path.join(self.dataset_path, 'pre-event',  self.dataset_list[idx] + '_pre_disaster' + self.suffix))
        post = self.loader(os.path.join(self.dataset_path, 'post-event', self.dataset_list[idx] + '_post_disaster' + self.suffix)) 
        post = np.expand_dims(post, axis=2)
        img = np.concatenate([pre, post], axis=2) 
        
        mask = self.loader(os.path.join(self.dataset_path, 'target', self.dataset_list[idx] + '_building_damage' + self.suffix))
        
        if self.transforms: 
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img.transpose(2, 0, 1), mask


class Trainer: 
    def __init__(self, args):
        self.args = args 

        # setup directory for saving model checkpoints 
        self.save_path = os.path.join(args.save_dir, args.run_name)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # setup tensorboard writer
        self.writer = SummaryWriter(
            log_dir =  os.path.join(self.save_path + '/tb_logs/')
        )

        # setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # setup model and training hyperparameters
        self.model = UNet(in_channels=4, n_classes=4)
        self.model.float()
        self.model.to(self.device)

        self.cl_criterion = nn.CrossEntropyLoss(reduction='mean')

        self.segm_criterion = smp.losses.LovaszLoss(mode='multiclass')

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        self.num_epochs = self.args.num_epochs

        self.train_iou_classes = MeanIoU(
            num_classes=4,
            per_class = True,  
            input_format = 'index' 
            ).to(self.device)

        self.val_iou_classes = MeanIoU(
            num_classes=4,
            per_class = True,
            input_format = 'index'
            ).to(self.device)

        # setup data loaders 
        self.train_transforms = A.Compose([
            A.Normalize(mean=[88.26, 83.83, 75.40, 57.39], std=[38.06, 31.84, 28.87, 24.43]),
            A.RandomCrop(width=self.args.crop_size, height=self.args.crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], additional_targets={'image': 'image', 'mask': 'mask'}, is_check_shapes=True)

        self.val_transforms = A.Compose([
           A.Normalize(mean=[88.26, 83.83, 75.40, 57.39], std=[38.06, 31.84, 28.87, 24.43]),
        ], additional_targets={'image': 'image', 'mask': 'mask'}, is_check_shapes=True)

        self.train_dataset = DFC25_TRAIN(
            dataset_path=args.dataset_train_path,
            dataset_file=args.train_file,
            loader=img_loader,
            transforms=self.train_transforms,
        )

        self.val_dataset = DFC25_VAL(
            dataset_path=args.dataset_val_path,
            dataset_file=args.val_file,
            loader=img_loader,
            transforms=self.val_transforms,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size // 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    def training(self): 

        best_epoch = 1 
        best_val_miou = 0.0

        for epoch in range(self.num_epochs): 
            
            # Training phase start
            self.model.train() 

            run_train_loss = 0.0 

            # setup tqdm for batch progress 
            pbar_train = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} - TRAIN', dynamic_ncols=True)

            for batch_idx, (inputs, targets) in enumerate(pbar_train): 
                # move inputs and outputs to same device as model
                inputs = inputs.to(self.device)
                targets = targets.long().to(self.device)

                # forward pass
                outputs = self.model(inputs) 

                # update step 
                self.optimizer.zero_grad()
                cl_loss = self.cl_criterion(outputs, targets)
                segm_loss = self.segm_criterion(outputs, targets)

                total_loss = cl_loss + 0.75*segm_loss
                
                total_loss.backward()
                self.optimizer.step()

                # log loss and accuracy 
                run_train_loss += total_loss.item()
                self.train_iou_classes.update(torch.argmax(outputs, axis=1), targets)

                # update pbar postfix during training (per batch)
                batch_miou = self.train_iou_classes.compute()
                batch_miou = batch_miou.cpu().numpy()
                pbar_train.set_postfix(train_loss=run_train_loss / (batch_idx + 1),
                                       train_miou=batch_miou)

            # End of train phase metric and loss logging
            run_train_loss /= len(self.train_loader)
            epoch_train_ious = self.train_iou_classes.compute().cpu().numpy()
            epoch_train_miou = epoch_train_ious.mean()
            epoch_train_iou_background = epoch_train_ious[0]
            epoch_train_iou_intact = epoch_train_ious[1]
            epoch_train_iou_damaged = epoch_train_ious[2]
            epoch_train_iou_destroyed = epoch_train_ious[3]

            self.writer.add_scalar('tLoss/train', run_train_loss, epoch)
            self.writer.add_scalar('mIoU/train', epoch_train_miou, epoch)
            self.writer.add_scalar('cIoU/train_background', epoch_train_iou_background, epoch)
            self.writer.add_scalar('cIoU/train_intact', epoch_train_iou_intact, epoch)
            self.writer.add_scalar('cIoU/train_damaged', epoch_train_iou_damaged, epoch)
            self.writer.add_scalar('cIoU/train_destroyed', epoch_train_iou_destroyed, epoch)

            self.writer.flush()

            # reset miou 
            self.train_iou_classes.reset()

            if (epoch + 1) % 1 == 0: 
                val_miou, val_iou_background, val_iou_intact, val_iou_damaged, val_iou_destroyed = self.validation(epoch)

                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))

                print(
                    f'Best validation mIoU: {best_val_miou} at epoch {best_epoch}\n'
                    f'Background IoU: {val_iou_background}\n'
                    f'Intact IoU: {val_iou_intact}\n'
                    f'Damaged IoU: {val_iou_damaged}\n'
                    f'Destroyed IoU: {val_iou_destroyed}\n'
                )
            self.train_iou_classes.reset()

    def validation(self, epoch):

        self.model.eval()

        pbar_val = tqdm(self.val_loader, 
                   desc=f'Epoch {epoch+1}/{self.num_epochs} - VAL', dynamic_ncols=True)

        with torch.no_grad():
            with profiler.profile(use_cuda=True) as prof:
                for batch_idx, (inputs, targets) in enumerate(pbar_val): 
                    if batch_idx > 1: break  # Profile only a few iterations
                    inputs, targets = inputs.to(self.device, non_blocking=True), targets.long().to(self.device, non_blocking=True)

                    # forward pass and loss computation
                    outputs = self.model(inputs)
        
                    preds = torch.argmax(outputs, dim=1)
                    self.val_iou_classes.update(preds, targets)

                # End of validation phase 
                epoch_val_ious = self.val_iou_classes.compute()
                val_miou = epoch_val_ious.mean()
                val_iou_background = epoch_val_ious[0].item()
                val_iou_intact = epoch_val_ious[1].item()
                val_iou_damaged = epoch_val_ious[2].item()
                val_iou_destroyed = epoch_val_ious[3].item()

                self.writer.add_scalar('mIoU/val', val_miou, epoch)
                self.writer.add_scalar('cIoU/val_background', val_iou_background, epoch)
                self.writer.add_scalar('cIoU/val_intact', val_iou_intact, epoch)
                self.writer.add_scalar('cIoU/val_damaged', val_iou_damaged, epoch)
                self.writer.add_scalar('cIoU/val_destroyed', val_iou_destroyed, epoch)

                self.writer.flush()

                print(
                    f'Validation mIoU: {val_miou}\n'
                    f'Background IoU: {val_iou_background}\n'
                    f'Intact IoU: {val_iou_intact}\n'
                    f'Damaged IoU: {val_iou_damaged}\n'
                    f'Destroyed IoU: {val_iou_destroyed}\n'
                )

            print(prof.key_averages().table(sort_by="cuda_time_total"))

            # reset validation accuracy 
            self.val_iou_classes.reset()

        return val_miou, val_iou_background, val_iou_intact, val_iou_damaged, val_iou_destroyed

print("train_baseline.py is being executed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_train_path', type=str, default='/home/gentleprotector/ubs_ws24/UBS-DFC25/data/dfc25_train')
    parser.add_argument('--train_file', type=str, default='/home/gentleprotector/ubs_ws24/UBS-DFC25/data/dfc25_train/train.txt')
    parser.add_argument('--dataset_val_path', type=str, default='/home/gentleprotector/ubs_ws24/UBS-DFC25/data/dfc25_train')
    parser.add_argument('--val_file', type=str, default='/home/gentleprotector/ubs_ws24/UBS-DFC25/data/dfc25_train/val.txt')

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop_size', type=int, default=256)

    parser.add_argument('--save_dir', type=str, default='/home/gentleprotector/ubs_ws24/UBS-DFC25/saved_models')
    parser.add_argument('--run_name', type=str, default=f'run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')



    args = parser.parse_args()

    trainer = Trainer(args)

    trainer.training() 

