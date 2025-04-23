import random
import torchvision
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import argparse
import os
from collections import OrderedDict
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yagmail
import time
import sys
from utils.metrics import iou_score,dice_score,f1_scorex
from utils.utils import AverageMeter, str2bool
from torch.utils.data import DataLoader
from skimage import io
from model.DRA_NET import DRA_net


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--name', default="UNET",
                        help='model name: Modified_UNET',choices=["DRA_net"])
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=6, type=int,
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=8, type=int)
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    # data
    parser.add_argument('--augmentation', type=str2bool, default=False, choices=[True, False])
    config = parser.parse_args()

    return config






class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transformers = None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.image_path = os.listdir(self.root_dir)
        self.label_path = os.listdir(self.label_dir)
        self.transformers = transformers
    def __getitem__(self, idx):
        img_name = self.image_path[idx]
        label_name = self.label_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        label_item_path = os.path.join(self.label_dir, label_name)
        image = io.imread(img_item_path)/256
        image = torch.from_numpy(image)
        label = io.imread(label_item_path)
        label = torch.from_numpy(label)
        return image,label
    def __len__(self):
        return len(self.image_path)


def train(train_loader, model, criterion, optimizer):
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'dice': AverageMeter(),
                      'f1-score': AverageMeter()
                      }
        model.train()
        pbar = tqdm(total=len(train_loader))
        for input, target in train_loader:
            input = input.float().cuda()
            target = target.long().cuda()
            b,h,w,c = input.size()
            input = input.reshape(b,c,h,w)
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_score(output, target)
            f1_score = f1_scorex(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1-score'].update(f1_score, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1-score', avg_meters['f1-score'].avg)])

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('f1-score', avg_meters['f1-score'].avg)
                            ])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'f1-score': AverageMeter()
                  }
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:

            input = input.float().cuda()
            target = target.long().cuda()
            b, h, w, c = input.size()
            input = input.reshape(b, c, h, w)
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_score(output, target)
            f1_score = f1_scorex(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1-score'].update(f1_score, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1-score', avg_meters['f1-score'].avg)])

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('f1-score', avg_meters['f1-score'].avg)
                            ])


def main():
    """
    创建储存最好模型、xml文件
    """
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)


    config = vars(parse_args())
    if config['augmentation'] == True:
        file_name = config['name'] + '_with_augmentation'
    else:
        file_name = config['name'] + '_base'
    os.makedirs('checpoint/newdata/{}'.format(file_name), exist_ok=True)
    print("Creating directory called",file_name)

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)
    #save configuration
    with open('checpoint/newdata/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)
    # cr20iterion = nn.CrossEntropyLoss(weight= torch.tensor([0,1,3,3,1.5,1,1]), reduction="mean").cuda()#
    criterion = nn.CrossEntropyLoss(weight=None, reduction="mean").cuda()  #
    # cudnn.benchmark = True
    print("=> creating model" )
    if config['name'] == 'DRA_net':
        model = DRA_net(3,7)

    else:
        raise ValueError("Wrong Parameters")
    model = model.cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) # 每经过step_size 个epoch，做一次学习率decay，以gamma值为缩小倍数。

############### Load the data path
    train_img = 'baixibao_mydata/train/image'
    train_label = 'baixibao_mydata/train/label'
    val_img = 'baixibao_mydata/valid/image'
    val_label = 'baixibao_mydata/valid/label'

    train_dataset = MyData(train_img, train_label)
    val_dataset = MyData(val_img, val_label)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    log= pd.DataFrame(index=[],columns= ['epoch','lr','loss','iou',"dice",'f1-score','val_loss','val_iou',"val_dice",'val_f1-score'])

    best_dice = 0
    trigger = 0

    for epoch in range(config['epochs']):

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        print('Training epoch [{}/{}], Training loss:{:.4f}, Training IOU:{:.4f}, Training DICE:{:.4f},Training f1-score:{:.4f}, Validation loss:{:.4f}, Validation IOU:{:.4f}, Validation DICE:{:.4f}, Validation f1-score:{:.4f}'.format(
            epoch + 1, config['epochs'], train_log['loss'], train_log['iou'],  train_log['dice'], train_log['f1-score'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['f1-score']))

        tmp = pd.Series([
            epoch,
            config['lr'],
            #train_log['lr_exp'],
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            train_log['f1-score'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice'],
            val_log['f1-score']
        ], index=['epoch', 'lr', 'loss', 'iou','dice','f1-score', 'val_loss', 'val_iou',"val_dice",'val_f1-score'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('checpoint/newdata/{}/log.csv'.format(file_name), index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'checpoint/newdata/{}/bestmodel_baixibao_{}_final.pth'.format(file_name,config["lr"]))
            best_dice = val_log['dice']
            print("=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
