# import wandb
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from importlib import import_module
from sklearn.metrics import f1_score
from collections import OrderedDict, Counter
from torch.utils.data.sampler import WeightedRandomSampler

import data_utils 
from data_utils import MaskDataset

import utils

import loss 
import models

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
 
def train(dataloader, model, criterion, optimizer, epoch):
    accumulation = 2
   
    loss_items = utils.AverageMeter()
    acc_items = utils.AverageMeter()
    model.train()
    for i, (images, labels) in enumerate(tqdm(dataloader, leave=False, total=100, ncols=80, desc=f"Epoch : {epoch:04d}")):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        batch = int(labels.size(0))
        outs = model(images)
        loss = criterion(outs, labels)
        loss.backward()
        
        if (i+1) % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        preds = torch.argmax(outs, dim=-1)
        acc = (preds==labels).sum().item()
        loss_items.update(loss.item(), batch)
        acc_items.update(acc/batch, batch)
    return OrderedDict([('Accuracy', acc_items.avg), 
                        ('Loss', loss_items.avg)])

def eval(dataloader, model, criterion, epoch):
    patience = 10
    counter = 0  
    val_loss_items = utils.AverageMeter()
    val_acc_items = utils.AverageMeter()
    pred_all, true_all = [], []
    model.eval()
    for images, labels in tqdm(dataloader, total=100, ncols=80, desc=f"Epoch {epoch:04d} Valid"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        batch = labels.size(0)
        outs = model(images)
        loss = criterion(outs, labels)
        preds = outs.argmax(dim=-1)
        acc = (preds==labels).sum().item()
        
        val_loss_items.update(loss.item(), batch)
        val_acc_items.update(acc/batch, batch)
        pred_all.append(preds.cpu())
        true_all.append(labels.cpu())
    pred = np.concatenate(pred_all)
    true = np.concatenate(true_all)
    f1 = f1_score(true, pred, average='macro')
    return OrderedDict([('Valid Accuracy', val_acc_items.avg), 
                        ('Valid Loss', val_loss_items.avg),
                        ('F1 score', f1)])       

def main(args):
    seed_everything(41)
    # if args.wandb_on:
    #     wandb.init(project="mask", 
    #                entity="0523kevin",
    #                name=f"{args.exp_name}",
    #                save_code=True,
    #                config = {'learning_rate':args.lr,
    #                          'optimizer':args.optimizer,
    #                          'model':args.model}
    #                )
    _logger = utils.get_logger(args.exp_name)
    utils.log_configs(args)
    
    _logger.info(f"Device : {DEVICE}")
    
    ckpt_dir = f'./results/{args.exp_name}/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
        
    transform = data_utils.init_transform(args.aug, p=1.0)
    transform_valid = data_utils.init_transform('valid', p=1.0)
    
    train_imgs, valid_imgs = data_utils.make_filelist(args.img_dir, val_size=0.2, stratify=True)
    
    train_dataset = MaskDataset(train_imgs, transform)
    valid_dataset = MaskDataset(valid_imgs, transform_valid)
    
    # weighted random sampler
    num_samples = len(train_dataset)
    train_labels = []
    for _, label in train_dataset:
        train_labels.append(label)
    class_cnts = Counter(train_labels)
    class_weights = [num_samples / class_cnts[i] for i in range(18)]
    weights = [class_weights[t] for t in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples)

    train_dataloader = data_utils.get_dataloader(train_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 drop_last=True,
                                                 sampler=sampler)
    valid_dataloader = data_utils.get_dataloader(valid_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 drop_last=True,
                                                 sampler=None)
    
    model = models.init_model(args.model, num_classes=18)
    criterion = loss.init_loss(args.loss)
    opt_module = getattr(import_module('torch.optim'), args.optimizer)
    optimizer = opt_module(filter(lambda x:x.requires_grad, model.parameters()), lr=args.lr)
    
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
        
    loss_curve = utils.CurvePlotter(title=f'{args.exp_name}', xlabel='Epoch', ylabel='Loss', i=1)
    acc_curve = utils.CurvePlotter(title=f'{args.exp_name}', xlabel='Epoch', ylabel='Accuracy', i=2)
    f1_curve = utils.CurvePlotter(title=f'{args.exp_name}', xlabel='Epoch', ylabel='F1_score', i=3)
    
    epoch_msg_header = (
        f"{'Epoch':^10}"
        f"{'Train Loss':^16}"
        f"{'Train Acc':^15}"
        f"{'Valid Loss':^16}"
        f"{'Valid Acc':^15}"
        f"{'Valid F1':^15}"
    )
    _logger.info(epoch_msg_header)
    epoch_msg_header = '\n' + '=' * 90 + '\n' + epoch_msg_header + '\n' + '=' * 90
    print(epoch_msg_header)
    
    best_f1 = args.best_f1
    for epoch in range(args.epochs):
        model.to(DEVICE)
        train_metrics = train(train_dataloader, model, criterion, optimizer, epoch)
        valid_metrics = eval(valid_dataloader, model, criterion, epoch)
        if scheduler:
            scheduler.step()
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([(f"train_{k}", v) for k, v in train_metrics.items()])
        metrics.update([(f"valid_{k}", v) for k, v in valid_metrics.items()])

        epoch_msg = (
            f"""{f'{epoch:04d}':^10}"""
            f"""{f"{metrics['train_loss']:.6f}":^16}"""
            f"""{f"{metrics['train_acc']:.4f}":^15}"""
            f"""{f"{metrics['valid_loss']:.6f}":^16}"""
            f"""{f"{metrics['valid_acc']:.4f}":^15}"""
            #
            f"""{f"{metrics['valid_f1']:.6f}":^16}"""
        )

        _logger.info(epoch_msg)
        print(epoch_msg)
        
        loss_curve.update_values('train_loss', metrics['train_loss'])
        loss_curve.update_values('valid_loss', metrics['valid_loss'])
        loss_curve.plot_learning_curve(label='train_loss')
        loss_curve.plot_learning_curve(label='valid_loss')
        loss_curve.save_fig(f'./results/{args.exp_name}/loss_curve.png')
        
        acc_curve.update_values('train_acc', metrics['train_acc'])
        acc_curve.update_values('valid_acc', metrics['valid_acc'])
        acc_curve.plot_learning_curve(label='train_acc')
        acc_curve.plot_learning_curve(label='valid_acc')
        acc_curve.save_fig(f'./results/{args.exp_name}/acc_curve.png')
        
        f1_curve.update_values('valid_f1', metrics['valid_f1'])
        f1_curve.plot_learning_curve(label='valid_f1')
        f1_curve.save_fig(f'./results/{args.exp_name}/f1_curve.png')
    
        # if not args.wandb_off:
        #     wandb.log({f"{args.target}_train_loss": metrics['train_loss'],
        #                f"{args.target}_train_acc": metrics['train_acc'],
        #                f"{args.target}_valid_acc": metrics['valid_acc'],
        #                f"{args.target}_valid_loss": metrics['valid_loss'],
        #                f"{args.target}_f1_score": metrics['valid_f1']})
        
        if args.save_best_ckpt:
            if valid_metrics['F1 score'] > best_f1:
                best_ckpt_path = os.path.join(ckpt_dir, f"best_epoch{epoch:04d}.pt")
                torch.save(model.state_dict(), best_ckpt_path)
                _logger.info(f"New best model is saved at {best_ckpt_path}")
                best_f1 = valid_metrics['F1 score']

        if epoch in args.save_ckpt_list:
            ckpt_dir = f'./results/{args.exp_name}/checkpoints'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch:04d}.pt')
            torch.save(model.state_dict(), ckpt_path)
            _logger.info(f'Checkpoint saved at {ckpt_path}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wandb_on', default=False, help='Do not log results in wandb')
    parser.add_argument('--save_ckpt', action='store_true', help='save checkpoint on best epoch')
    parser.add_argument('--save_ckpt_list', default=[1, 5, 10, 15, 20, 25, 30], help='save checkpoint on best epoch')
    parser.add_argument('--best_f1', type=int, default=0, help="If you reload training, set the best_f1 for last train's best f1 score")
    parser.add_argument('--exp_name', type=str, default='exp1', help='Experiment name')
    parser.add_argument('--img_dir', type=str, default='/home/appler/cv/classification/classification/data/train/images', help='img directory')
    parser.add_argument('--aug', type=str, default='gray_crop', help='Data augmentation for train dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs.')
    parser.add_argument('--model', type=str, default='resnet34', help='Name of the model to train')
    parser.add_argument('--loss', type=str, default='ce', help='Loss Function')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--scheduler', type=bool, default=True, help='Use or not CosineAnnealingLR scheduler')
    parser.add_argument('--seed', type=int, default=42, help='Random seed setting')
    args = parser.parse_args()
    main(args)
