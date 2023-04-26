from __future__ import print_function

import argparse
import gc
import os
import shutil
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timm.loss import LabelSmoothingCrossEntropy
import models.cifar as models
from utils.bypass_bn import enable_running_stats, disable_running_stats
from utils.logger import Logger, savefig
from utils.misc import AverageMeter, mkdir_p
from utils.eval import accuracy
import torch.nn.functional as F
from queue import Queue
import numpy as np
import math

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--smoothing", default=0.0, type=float, help="Label smoothing.")

parser.add_argument("--T", type=float, default=1.5, help='Distillation temperature (hyperparameter of SMC-2)')
parser.add_argument("--alpha", type=float, default=0.9, help='Weighting ratio (hyperparameter of SMC-2)')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
use_cuda = torch.cuda.is_available()
# Seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
cudnn.benchmark = False
cudnn.deterministic = True
# save path
args.checkpoint = "checkpoints_modify/{}/SMC-2_seed{}_T{}_a{}_m{}".format(
    args.dataset, args.manualSeed, args.T, args.alpha, args.checkpoint)

total_step = math.ceil((50000.0 / args.train_batch)) * args.epochs
best_acc = 0.
best_epoch = 0
now_step = 0

def test(testloader, model, criterion, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0].item(), inputs.size(0))
            top5.update(prec5[0].item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return losses.avg, top1.avg, top5.avg, batch_time.sum

def train(train_A_loader, train_B_loader, model, criterion, optimizer, epoch, use_cuda, B_loader, pre_outs):
    global now_step
    global total_step
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ce_losses = AverageMeter()
    dml_losses = AverageMeter()
    A_top1 = AverageMeter()

    end = time.time()

    if B_loader is None:
        B_loader = enumerate(train_B_loader)
        index, (B_imgs, B_label) = next(B_loader)

    for batch_idx, (inputs, targets) in enumerate(train_A_loader):
        a = args.alpha * (1 - 0.5 * (1 + math.cos(math.pi * now_step / total_step)))
        data_time.update(time.time() - end)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Obtain B-channel data
        try:
            index, (B_imgs, B_label) = next(B_loader)
        except StopIteration:
            B_loader = enumerate(train_B_loader)
            index, (B_imgs, B_label) = next(B_loader)

        # Obtaining soft labels for B-channel data
        enable_running_stats(model)
        with torch.no_grad():
            if use_cuda:
                B_imgs = B_imgs.cuda()
            B_out = model(B_imgs).detach_()
            pre_outs.put(B_out)
        B_imgs, B_out, B_label = None, None, None

        # Calculating the loss function
        disable_running_stats(model)
        outputs = model(inputs)
        if pre_outs.qsize() >= 2:
            pre_out = pre_outs.get()
            if use_cuda:
                pre_out = pre_out.cuda()
            ce_loss = criterion(outputs, targets)
            dml_loss = (
                F.kl_div(
                    F.log_softmax(outputs / args.T, dim=1),
                    F.softmax(pre_out / args.T, dim=1),
                    reduction="batchmean",
                )
            )
            pre_out = None
            loss = (1 - a) * ce_loss + a * dml_loss
            with torch.no_grad():
                ce_losses.update((1 - a) * ce_loss.item(), targets.size(0))
                dml_losses.update(a * dml_loss.item(), targets.size(0))
        else:
            loss = criterion(outputs, targets)
            with torch.no_grad():
                ce_losses.update(loss.item(), targets.size(0))

        with torch.no_grad():
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))
            A_top1.update(prec1[0].item(), targets.size(0))

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        with torch.no_grad():
            batch_time.update(time.time() - end)
            end = time.time()

        now_step += 1

    # Clean up memory
    gc.collect()
    return ce_losses.avg, dml_losses.avg, A_top1.avg, batch_time.sum, B_loader, pre_outs

def main():
    global best_acc
    global best_epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        args.length = 16
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
        args.length = 8
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_A_set = dataloader(root='./data', train=True, download=True, transform=transform_train)
    # Set seeds so that the data in each channel is sorted consistently
    torch.manual_seed(args.manualSeed)
    g_A = torch.Generator()
    train_A_loader = data.DataLoader(
        train_A_set, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, generator=g_A)

    train_B_set = dataloader(root='./data', train=True, download=True, transform=transform_train)
    torch.manual_seed(args.manualSeed)
    g_B = torch.Generator()
    train_B_loader = data.DataLoader(
        train_B_set, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, generator=g_B)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    print(args.arch)
    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.csv'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.csv'), title=title)
        logger.set_names(['Learning Rate', 'Train ceLoss', 'Train dmlLoss', 'Train Top1 acc', 'Train time',
                          'Valid Loss', 'Valid Top1 acc', 'Valid Top5 acc', 'Valid time'])

    B_loader, pre_outs = None, Queue()
    for epoch in range(0, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, scheduler.get_last_lr()[0]))
        # train
        train_celoss, train_dmlloss, train_top1_acc, train_time, B_loader, pre_outs = train(
            train_A_loader, train_B_loader, model, criterion, optimizer, epoch, use_cuda, B_loader, pre_outs)
        # test
        test_loss, test_top1_acc, test_top5_acc, test_time = test(
            testloader, model, criterion,  use_cuda)
        # append logger file log
        logger.append(
            [scheduler.get_last_lr()[0], train_celoss, train_dmlloss, train_top1_acc, train_time,
             test_loss, test_top1_acc, test_top5_acc, test_time])

        # save model
        is_best = test_top1_acc > best_acc
        best_acc = max(test_top1_acc, best_acc)
        if is_best:
            # save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': model.state_dict(),
            #         'acc': test_top1_acc,
            #         'best_acc': best_acc,
            #         'optimizer' : optimizer.state_dict(),
            #     }, is_best, checkpoint=args.checkpoint)
            best_epoch = epoch + 1
        scheduler.step()

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:', best_acc)
    print('Best epoch:', best_epoch)

if __name__ == '__main__':
    # args.arch = 'vgg19_bn'
    # args.dataset = 'cifar100'
    # args.epochs = 200
    # args.wd = 5e-4
    # args.manualSeed = 9
    # args.checkpoint = 'vgg19'
    main()


