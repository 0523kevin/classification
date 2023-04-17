"""
train.py
"""

import os
from argparse import ArgumentParser
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch
import timm

import utils
import data_utils
import models

import wandb


def train(model, dataloader, criterion, optimizer, device, epoch):
    loss = utils.AverageMeter()
    acc = utils.AverageMeter()

    model.train()

    for inputs, targets in tqdm(dataloader, leave=False, total=100, ncols=80, desc=f'Epoch {epoch:04d} Train'):
        inputs, targets = inputs.to(device), targets.to(device)
        b = targets.size(0)  # batch size
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss_ = criterion(outputs, targets)
        loss_.backward()
        optimizer.step()
        loss.update(loss_.item(), b)

        preds = outputs.argmax(dim=1)
        acc.update(torch.sum(preds==targets).item() / b, b)

    return OrderedDict([('loss', loss.avg), ('acc', acc.avg)])


def eval(model, dataloader, criterion, device, epoch):
    loss = utils.AverageMeter()
    acc = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, leave=False, total=100, ncols=80, desc=f'Epoch {epoch:04d} Valid'):
            inputs, targets = inputs.to(device), targets.to(device)
            b = targets.size(0)  # batch size
            
            outputs = model(inputs)
            
            loss_ = criterion(outputs, targets)
            loss.update(loss_.item(), b)

            preds = outputs.argmax(dim=1)
            acc.update(torch.sum(preds==targets).item() / b, b)

    return OrderedDict([('loss', loss.avg), ('acc', acc.avg)])



def main(args):
    _logger = utils.get_logger(args.exp_name)
    utils.log_configs(args)

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info(f'Device: {DEVICE}')

    utils.seed_everything(args.seed)

    ##################### wandb #######################
    if not args.wandb_off:
        wandb.init(
            project="mask-classification",
            entity="5pencv",
            name=f"{args.exp_name}",
            save_code=True,

            config={
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "model": args.model,
                "loss": "CrossEntropyLoss",
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "target": args.target,
                "augmentation": args.aug,
            }
        )
    ##################### ##### #######################

    # transform = data_utils.transform_dict[args.aug] if args.aug in data_utils.transform_dict.keys() else None

    transform = data_utils.init_transform(args.aug, args.p)

    train_image_files = data_utils.generate_file_list(args.datadir, val_split=args.val_split, train=True)
    valid_image_files = data_utils.generate_file_list(args.datadir, val_split=args.val_split, train=False)
    
    train_dataset = data_utils.MaskDataset(train_image_files, args.target, group_age=True, transform=transform)
    valid_dataset = data_utils.MaskDataset(valid_image_files, args.target, group_age=True, transform=None)
    
    train_dataloader = data_utils.get_dataloader(train_dataset,
                                                args.batch_size,
                                                shuffle=args.shuffle_off,
                                                drop_last=True,
                                                )
    valid_dataloader = data_utils.get_dataloader(valid_dataset,
                                                args.valid_batch_size,
                                                shuffle=False,
                                                drop_last=True,
                                                )

    # model_dict = models.init_model_dict()

    # if args.model in model_dict.keys():
    #     model = getattr(models, model_dict[args.model])(args.n_class)
    # else:
    #    raise ValueError(f"'{args.model}' not implemented!")

    # model = timm.create_model('resnet34', pretrained=True, num_classes=args.n_class)
    # model = timm.create_model('convnext_small', pretrained=True, num_classes=args.n_class)

    model = models.init_model(args.model, args.n_class)
    model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.optimizer](
        model.parameters(), lr=args.lr
    )

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None


    loss_curve = utils.CurvePlotter(title=f'{args.exp_name}', xlabel='Epoch', ylabel='Loss', i=1)
    acc_curve = utils.CurvePlotter(title=f'{args.exp_name}', xlabel='Epoch', ylabel='Accuracy', i=2)

    epoch_msg_header = (
        f"{'Epoch':^10}"
        f"{'Train Loss':^16}"
        f"{'Train Acc':^15}"
        f"{'Valid Loss':^16}"
        f"{'Valid Acc':^15}"
    )
    _logger.info(epoch_msg_header)
    epoch_msg_header = '\n' + '=' * 75 + '\n' + epoch_msg_header + '\n' + '=' * 75
    print(epoch_msg_header)

    for epoch in range(1, args.epochs+1):
        train_metrics = train(model, train_dataloader, criterion, optimizer, DEVICE, epoch)
        valid_metrics = eval(model, valid_dataloader, criterion, DEVICE, epoch)
    
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([(f'train_{k}', v) for k, v in train_metrics.items()])
        metrics.update([(f'valid_{k}', v) for k, v in valid_metrics.items()])

        epoch_msg = (
            f"""{f'{epoch:04d}':^10}"""
            f"""{f"{metrics['train_loss']:.6f}":^16}"""
            f"""{f"{metrics['train_acc']:.4f}":^15}"""
            f"""{f"{metrics['valid_loss']:.6f}":^16}"""
            f"""{f"{metrics['valid_acc']:.4f}":^15}"""
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

        if not args.wandb_off:
            wandb.log({
                f"{args.target}_train_loss": metrics['train_loss'],
                f"{args.target}_train_acc": metrics['train_acc'],
                f"{args.target}_valid_acc": metrics['valid_acc'],
                f"{args.target}_valid_loss": metrics['valid_loss'],
                })

        if scheduler:
            scheduler.step()
    
        if args.save_ckpt:
            ckpt_dir = f'./results/{args.exp_name}/checkpoints'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch:04d}.pt')
            torch.save(model.state_dict(), ckpt_path)
            _logger.info(f'Checkpoint saved at {ckpt_path}')


if __name__ == "__main__":
    parser = ArgumentParser()

    # experiment and log settings
    parser.add_argument('--wandb_off', action='store_true', help='Do not log results in wandb')
    parser.add_argument('--save_ckpt', action='store_false', help="Save checkpoint at the end of every epoch.")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name")

    # datasets
    parser.add_argument('--datadir', '--data_dir', type=str, default='input/data/train/images', help='Data directory.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio. Set zero to use all data for training.')
    parser.add_argument('--shuffle_off', action='store_false', help="Do not shuffle train dataset.")
    parser.add_argument('--aug', type=str, default='None', help='Data augmentation.')
    parser.add_argument('--p', type=float, default=1, help='Possibility')

    # model configs
    parser.add_argument('--model', type=str, help='Name of the model to train.')
    parser.add_argument('--n_class', type=int, default=18, help="The number of classes.")
    parser.add_argument('--target', type=str, default='all', choices=['all', 'age', 'gender', 'mask'], help='Target label to predict.')

    # hyper-parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--valid_batch_size', type=int, default=100, help='input batch size for validing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam'], help='Optimizer.')
    parser.add_argument('--scheduler', action='store_true', help='Use CosineAnnealingLR scheduler.')

    # system
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    main(args)