"""
Training script for ImageNet
Copyright (c) Wei YANG, 2017
"""
from __future__ import print_function

import argparse
import math
import os
import shutil
import time
import random
from datetime import datetime
import json
import matplotlib.pyplot as plt
from torcheval.metrics import BinaryAUROC 

from torchvision.transforms.transforms import (
    ColorJitter,
    RandomAffine,
    RandomGrayscale,
    RandomPerspective,
)
from utils.focal import FocalLoss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as tf
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F 
import models
from math import cos, pi
from models.resnet import fc_block
from models.resnet import MaskedLoss
from tqdm import tqdm

from celeba import CelebA, LFW
from utils import (
    Bar,
    Logger,
    AverageMeter,
    accuracy,
    mkdir_p,
    savefig,
    accuracy_bce,
    stat,
)
from tensorboardX import SummaryWriter


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


# Parse arguments
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("-d", "--data", default="path to dataset", type=str)
parser.add_argument("-dl", "--data_lfw", default="path to lfw dataset", type=str)
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
# Optimization options
parser.add_argument(
    "--epochs", default=30, type=int, metavar="N", help="number of total epochs to run"
)

parser.add_argument(
    "-rs", action="store_true", help="ReweightedRandomSampler",
)

parser.add_argument(
    "-lw", action="store_true", help="Reverse Sample Count CE Weight",
)

parser.add_argument(
    "-fc", "--focal", action="store_true", help="Reverse Sample Count CE Weight",
)

parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--train-batch",
    default=64,
    type=int,
    metavar="N",
    help="train batchsize (default: 256)",
)
parser.add_argument(
    "--test-batch",
    default=320,
    type=int,
    metavar="N",
    help="test batchsize (default: 320)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--lr-decay", type=str, default="cos", help="mode for learning rate decay"
)

parser.add_argument("--sampler", type=str, default="uniform", help="data sampler")

parser.add_argument(
    "--step", type=int, default=20, help="interval for learning rate decay in step mode"
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[150, 225],
    help="decrease learning rate at these epochs.",
)
parser.add_argument(
    "--turning-point",
    type=int,
    default=100,
    help="epoch number from linear to exponential decay mode",
)
parser.add_argument(
    "--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule."
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
# Checkpoints
parser.add_argument(
    "-c",
    "--checkpoint",
    default="checkpoints",
    type=str,
    metavar="PATH",
    help="path to save checkpoint (default: checkpoints)",
)

parser.add_argument(
    "--ft", action="store_true", help="fine tune on Balance Class Sampler",
)

parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
# Architecture
parser.add_argument(
    "--cardinality", type=int, default=32, help="ResNeXt model cardinality (group)."
)
parser.add_argument(
    "--base-width",
    type=int,
    default=4,
    help="ResNeXt model base width (number of channels in each group).",
)
parser.add_argument("--groups", type=int, default=3, help="ShuffleNet model groups")
# Miscs
parser.add_argument("--manual-seed", type=int, help="manual seed")
parser.add_argument(
    "-e", "--evaluate", action="store_true", help="evaluate model on test set",
)
parser.add_argument(
    "-v", "--validate", action="store_true", help="evaluate model on validation set",
)
parser.add_argument(
    "-el", "--evaluate_lfw", action="store_true", help="evaluate model on lfw set",
)
parser.add_argument(
    "-pt",
    "--pretrained",
    dest="pretrained",
    action="store_true",
    help="use pre-trained model",
)
parser.add_argument(
    "--world-size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="gloo", type=str, help="distributed backend"
)
# Device options
parser.add_argument(
    "--gpu-id", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)
# parser.add_argument(
#     "--name", default='radial_based_mask_path1_align_with_threshold', type=str
# )
parser.add_argument(
    "--no-amp",
    action="store_true",
    help="run entirely in FP32 (disable automatic mixed precision)",
)


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Use CUDA
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    # Random seed
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # Get current timestamp for log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = "v7-fp32_celebA_resnet_50"
    
    # Create a more organized log directory structure with timestamp
    log_dir = os.path.join("log", "celebA-resnet50", f"{experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Create train and val subdirectories
    train_img_dir = os.path.join(log_dir, "train")
    val_img_dir = os.path.join(log_dir, "val")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    #if hyperparams is None: 
    hyperparams = default_hyperparams(log_dir)

    # create model
    if args.resume == "" and args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith("resnext"):
        model = models.__dict__[args.arch](
            baseWidth=args.base_width, cardinality=args.cardinality,
        )
    elif args.arch.startswith("shufflenet"):
        model = models.__dict__[args.arch](groups=args.groups)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False)

    if args.ft:
        for param in model.parameters():
            param.requires_grad = False
        classifier_numbers = model.num_attributes
        # newly constructed classifiers have requires_grad=True
        for i in range(classifier_numbers):
            setattr(
                model,
                "classifier" + str(i).zfill(2),
                nn.Sequential(fc_block(512, 256), nn.Linear(256, 1)),
            )

        args.sampler = "balance"

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
        )

    if not args.distributed:
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            # model = torch.nn.DataParallel(model).cuda()
            model.cuda()
    else:
        model.cuda()
        # model = torch.nn.parallel.DistributedDataParallel(model)

    # optionally resume from a checkpoint
    title = "CelebA-" + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(  # statistics from CelebA TrainSet
        mean=[0.5084, 0.4287, 0.3879], std=[0.2656, 0.2451, 0.2419]
    )
    print("=> using {} sampler to load data.".format(args.sampler))

    train_dataset = CelebA(
        args.data,
        "train_attr_list.txt",
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0)),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(),
            ]
        ),
        sampler=args.sampler,
    )

    train_sample_prob = train_dataset._class_sample_prob()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        if args.rs:
            train_sampler = WeightedRandomSampler(
                1 / train_sample_prob, len(train_dataset)
            )
        else:
            train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        CelebA(
            args.data,
            "val_attr_list.txt",
            transforms.Compose(
                [transforms.Resize(size=(256, 256)), transforms.ToTensor(), normalize,]
            ),
        ),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        CelebA(
            args.data,
            "test_attr_list.txt",
            transforms.Compose(
                [transforms.Resize(size=(256, 256)), transforms.ToTensor(), normalize,]
            ),
        ),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    lfw_test_loader = torch.utils.data.DataLoader(
        LFW(
            args.data_lfw, transforms.Compose([transforms.ToTensor(), normalize,]),
        ),  # celebA mean variance
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # define loss function (criterion) and optimizer
    if args.lw:  # loss weight
        print("=> loading CE loss_weight")
        criterion = nn.BCEWithLogitsLoss(
            reduction="mean", weight=1 / torch.sqrt(train_sample_prob)
        ).cuda()
    else:
        # criterion = nn.CrossEntropyLoss().cuda()
        criterion = MaskedLoss(
            lambda_mask=hyperparams['initial_lambda_mask'], 
            lambda_fully_masked=hyperparams['initial_lambda_fully_masked'],
            lambda_smoothness=hyperparams['initial_lambda_smoothness'],
            dynamic_masked_weight_min=hyperparams['dynamic_masked_weight_min'],
            dynamic_masked_weight_max=hyperparams['dynamic_masked_weight_max'],
            lambda_alignment=hyperparams['initial_lambda_alignment'],
            upper_mask_level_threshold=0.8,
            train_sample_prob=train_sample_prob,
        )

    if args.focal:
        print("=> using focal loss")
        criterion = FocalLoss(criterion, balance_param=5)

    unet_params      = list(model.mask_generator.parameters())
    resnet_params  = [p for n, p in model.named_parameters()
                        if not n.startswith("mask_generator.")]


    # Initially freeze U-Net weights if start_epoch_unet > 0
    if hyperparams["start_epoch_unet"] > 0:
        for param in model.mask_generator.parameters():
            param.requires_grad = False
        print(f"U-Net weights frozen until epoch {hyperparams['start_epoch_unet']}")
    
    print("=> using wd {}".format(args.weight_decay))
    unet_optimizer = torch.optim.AdamW(
        unet_params,
        lr=hyperparams['unet_learning_rate'],                   # same LR you passed to SGD
        betas=(0.9, 0.999),           # default Adam moments; adjust if you like
        weight_decay=args.weight_decay,
        eps=1e-8                      # default numerical stability term
    )
    
    resnet_optimizer = torch.optim.AdamW(
        resnet_params,
        lr=hyperparams['resnet_learning_rate'],
        betas=(0.9, 0.999),           # default Adam moments; adjust if you like
        weight_decay=args.weight_decay,
        eps=1e-8   
    )

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            resnet_optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(
                os.path.join(args.checkpoint, "log.txt"), title=title, resume=True
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, "log.txt"), title=title)
        logger.set_names(
            [
                "Learning Rate",
                "Train Loss",
                "Valid Loss",
                "Train Acc.",
                "Valid Acc.",
                "LFW Loss.",
                "LFW Acc.",
            ]
        )

    if args.evaluate:  # TODO
        validate(test_loader, model, criterion)
        # stat(train_loader)
        return
    if args.validate:  # TODO
        validate(val_loader, model, criterion)
        # stat(train_loader)
        return

    if args.evaluate_lfw:
        validate(val_loader, model, criterion)
        validate(lfw_test_loader, model, criterion)
        return

    classes = (
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
        "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
        "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
        "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
        "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
        "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
        "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
        "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
        "Wearing_Necklace", "Wearing_Necktie", "Young"
    )

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, "logs"))


    # Initialize metrics tracking
    best_val_masked_acc = 0
    metrics_header = "epoch,lambda_mask,lambda_fully_masked,lambda_smoothness,lambda_alignment,dynamic_weight,train_total_loss,train_masked_loss,train_unmasked_loss,train_masking_loss,train_fully_masked_loss,train_smoothness_loss,train_alignment_loss,train_alignment_divergence,train_mask_mean,train_masked_pixels_pct,train_binary_mask_mean,train_fully_masked_pct,train_masked_acc,train_unmasked_acc,val_total_loss,val_masked_loss,val_unmasked_loss,val_mask_mean,val_masked_pixels_pct,val_binary_mask_mean,val_fully_masked_pct,val_masked_acc,val_unmasked_acc,val_masking_loss,val_fully_masked_loss,val_smoothness_loss,val_alignment_loss,val_alignment_divergence,val_dynamic_weight\n"
    
    # Simplified to just track metrics in a CSV file
    with open(os.path.join(log_dir, "training_metrics.csv"), "w") as f:
        f.write(metrics_header)
    
    
    # preview results before training
    epoch_train_dir = os.path.join(train_img_dir, f"epoch_0")
    epoch_val_dir = os.path.join(val_img_dir, f"epoch_0")
    os.makedirs(epoch_train_dir, exist_ok=True)
    os.makedirs(epoch_val_dir, exist_ok=True)
    print("visualizing training samples BEFORE any training")
    visualize_dataset_samples(train_loader, 0, 8, "train", model, classes, epoch_train_dir)
    unet_lr = hyperparams['unet_learning_rate']
    resnet_lr = hyperparams['resnet_learning_rate']
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        resnet_lr = adjust_learning_rate(resnet_optimizer, epoch, hyperparams['resnet_learning_rate'])
        
        

         # Create epoch-specific directories
        epoch_train_dir = os.path.join(train_img_dir, f"epoch_{epoch+1}")
        epoch_val_dir = os.path.join(val_img_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_train_dir, exist_ok=True)
        os.makedirs(epoch_val_dir, exist_ok=True)

        print("\nEpoch: [%d | %d] Resnet LR: %f  UNet LR: %f" % (epoch + 1, args.epochs, resnet_lr, unet_lr))

        if epoch == hyperparams["start_epoch_unet"]:
            for param in model.mask_generator.parameters():
                param.requires_grad = True
            print(f"Epoch {epoch+1}: Unfreezing U-Net weights")
            unet_optimizer = torch.optim.AdamW(
                unet_params,
                lr=unet_lr,                   # same LR you passed to SGD
                betas=(0.9, 0.999),           # default Adam moments; adjust if you like
                weight_decay=args.weight_decay,
                eps=1e-8                      # default numerical stability term
            )
        unet_lr = adjust_learning_rate(unet_optimizer, epoch, hyperparams['unet_learning_rate'])
        # Update lambda values based on current epoch, considering start epochs
        # Mask loss
        
        if epoch >= hyperparams['start_epoch_mask']:
            
            # Calculate progress ratio only for epochs after start_epoch
            progress_ratio = (epoch - hyperparams['start_epoch_mask']) / max(1, (hyperparams['num_epochs'] - 1 - hyperparams['start_epoch_mask']))
            current_lambda_mask = hyperparams['initial_lambda_mask'] + (hyperparams['final_lambda_mask'] - hyperparams['initial_lambda_mask']) * progress_ratio * progress_ratio
        else:
            current_lambda_mask = 0.0  # No mask loss before start epoch
        criterion.lambda_mask = current_lambda_mask
        
        # Binary mask loss
        if epoch >= hyperparams['start_epoch_fully_masked']:
            progress_ratio = (epoch - hyperparams['start_epoch_fully_masked']) / max(1, (hyperparams['num_epochs'] - 1 - hyperparams['start_epoch_fully_masked']))
            current_lambda_fully_masked = hyperparams['initial_lambda_fully_masked'] + (hyperparams['final_lambda_fully_masked'] - hyperparams['initial_lambda_fully_masked']) * progress_ratio
        else:
            current_lambda_fully_masked = 0.0  # No binary mask loss before start epoch
        criterion.lambda_fully_masked = current_lambda_fully_masked
        
        # Smoothness loss
        if epoch >= hyperparams['start_epoch_smoothness']:
            progress_ratio = (epoch - hyperparams['start_epoch_smoothness']) / max(1, (hyperparams['num_epochs'] - 1 - hyperparams['start_epoch_smoothness']))
            current_lambda_smoothness = hyperparams['initial_lambda_smoothness'] + (hyperparams['final_lambda_smoothness'] - hyperparams['initial_lambda_smoothness']) * progress_ratio
        else:
            current_lambda_smoothness = 0.0  # No smoothness loss before start epoch
        criterion.lambda_smoothness = current_lambda_smoothness
        
        # Alignment loss
        if epoch >= hyperparams['start_epoch_alignment']:
            progress_ratio = (epoch - hyperparams['start_epoch_alignment']) / max(1, (hyperparams['num_epochs'] - 1 - hyperparams['start_epoch_alignment']))
            current_lambda_alignment = hyperparams['initial_lambda_alignment'] + (hyperparams['final_lambda_alignment'] - hyperparams['initial_lambda_alignment']) * progress_ratio
        else:
            current_lambda_alignment = 0.0  # No alignment loss before start epoch
        criterion.lambda_alignment = current_lambda_alignment
        
        # Dynamic weighting (optional - you can decide if this should honor start epoch too)
        if epoch < hyperparams['start_epoch_dynamic_weight']:
            # Force weight to 1.0 before start epoch (no dynamic weighting)
            criterion.dynamic_masked_weight_min = 1.0
            criterion.dynamic_masked_weight_max = 1.0
        else:
            # Restore original dynamic weight range after start epoch
            criterion.dynamic_masked_weight_min = hyperparams['dynamic_masked_weight_min']
            criterion.dynamic_masked_weight_max = hyperparams['dynamic_masked_weight_max']
        
        train_metrics = {
            'total_loss': 0.0,
            'masked_loss': 0.0,
            'resnet_loss' : 0.0,
            'unet_loss' : 0.0,
            'unmasked_loss': 0.0,
            'masking_loss': 0.0,
            'fully_masked_loss': 0.0,
            'smoothness_loss': 0.0,
            'alignment_loss': 0.0,  # Add alignment loss
            'alignment_divergence': 0.0,  # Add raw divergence value
            'mask_mean': 0.0,
            'binary_mask_mean': 0.0,
            'fully_masked_pct': 0.0,
            'masked_acc': 0.0,
            'unmasked_acc': 0.0,
            'dynamic_weight': 0.0,
            'masked_binauroc' : 0.0,
            'unmasked_binauroc' : 0.0
        }
        

        # train for one epoch
        train_metrics = train(train_loader, model, criterion, unet_optimizer, unet_params, resnet_optimizer, resnet_params, epoch, log_dir, train_metrics, hyperparams)

        # evaluate on validation set -- TO IMPLEMENT
        # val_loss, prec1 = validate(val_loader, model, criterion)
        val_loss, prec1 = -999, -999

        # # evaluate on lfw
        # lfw_loss, lfw_prec1 = validate(lfw_test_loader, model, criterion)
        lfw_loss, lfw_prec1 = -999, -999
        # append logger file
        logger.append([unet_lr, train_metrics['total_loss'], val_loss, train_metrics['masked_acc'], prec1, lfw_loss, lfw_prec1])

        # tensorboardX
        writer.add_scalar("learning rate", unet_lr, epoch + 1)
        writer.add_scalars(
            "loss",
            {
                "train total loss": train_metrics['total_loss'],
                "validation loss": val_loss,
                "lfw loss": lfw_loss,
            },
            epoch + 1,
        )
        writer.add_scalars(
            "accuracy",
            {
                "train mask accuracy": train_metrics['masked_acc'],
                "validation accuracy": prec1,
                "lfw accuracy": lfw_prec1,
            },
            epoch + 1,
        )
        # for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)
        
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        # save_checkpoint(
        #     {
        #         "epoch": epoch + 1,
        #         "arch": args.arch,
        #         "state_dict": model.state_dict(),
        #         "best_prec1": best_prec1,
        #         "optimizer": optimizer.state_dict(),
        #     },
        #     is_best,
        #     checkpoint=args.checkpoint,
        # )
        batch_count = len(train_loader)
        for key in train_metrics:
            if key not in ['masked_binauroc', 'unmasked_binauroc']:
                train_metrics[key] /= batch_count
        
        # Calculate masked pixel percentage for training - use mask_mean directly (1 = masked)
        train_masked_pixels_pct = train_metrics['mask_mean'] * 100
        
        # Run validation
        val_metrics = validate_model(val_loader, model, criterion)
        
        # Improved terminal output with more details
        print(f"Epoch {epoch+1}/{hyperparams['num_epochs']}, λ_mask={current_lambda_mask:.2f}, λ_fully_masked={current_lambda_fully_masked:.4f}, λ_smoothness={current_lambda_smoothness:.4f}, λ_alignment={current_lambda_alignment:.4f}")
        print(f"  TRAIN - ResNet LR: {resnet_lr:.4f}, UNet LR: {unet_lr:.4f}")
        print(f"  TRAIN - Dynamic Weight: {train_metrics['dynamic_weight']:.2f}, Val: {val_metrics['dynamic_weight']:.2f}")
        print(f"  TRAIN - Unmasked: Loss={train_metrics['unmasked_loss']:.4f}, Acc={train_metrics['unmasked_acc']:.2f}%")
        print(f"  TRAIN - Masked:   Loss={train_metrics['masked_loss']:.4f}, Acc={train_metrics['masked_acc']:.2f}%, Mask Loss={train_metrics['masking_loss']:.4f}")
        print(f"  TRAIN - Unmasked BinAUROC = {train_metrics['unmasked_binauroc'].mean():.4f}, Masked BinAUROC = {train_metrics['masked_binauroc'].mean():.4f}%")
        print(f"  TRAIN - Masking:  Soft={train_masked_pixels_pct:.1f}%, Binary={train_metrics['fully_masked_pct']:.1f}%, Smoothness={train_metrics['smoothness_loss']:.4f}")
        print(f"  TRAIN - Binary Loss: Raw={train_metrics['fully_masked_loss']:.4f}, Weighted={train_metrics['fully_masked_loss']:.4f}")
        print(f"  TRAIN - Alignment: Loss={train_metrics['alignment_loss']:.4f}, Divergence={train_metrics['alignment_divergence']:.4f}")  # Add alignment metrics
        print(f"  TRAIN - Resnet Loss={train_metrics['resnet_loss']}, UNet Loss={train_metrics['unet_loss']}")  # Add resnset loss
        print(f"  VAL   - Unmasked: Loss={val_metrics['unmasked_loss']:.4f}, Acc={val_metrics['unmasked_acc']:.2f}%")
        print(f"  VAL   - Masked:   Loss={val_metrics['masked_loss']:.4f}, Acc={val_metrics['masked_acc']:.2f}%, Mask Loss={val_metrics['masking_loss']:.4f}")
        print(f"  VAL   - Masking:  Soft={val_metrics['masked_pixels_pct']:.1f}%, Binary={val_metrics['fully_masked_pct']:.1f}%, Smoothness={val_metrics['smoothness_loss']:.4f}")
        print(f"  VAL   - Binary Loss: Raw={val_metrics['fully_masked_loss']:.4f}, Weighted={val_metrics['fully_masked_loss']:.4f}")
        print(f"  VAL   - Alignment: Loss={val_metrics['alignment_loss']:.4f}, Divergence={val_metrics['alignment_divergence']:.4f}")  # Add alignment metrics
        print(f"  VAL   - Unmasked BinAUROC = {val_metrics['unmasked_binauroc'].mean():.4f}, Masked BinAUROC = {val_metrics['masked_binauroc'].mean():.2f}%")
        
        # Check if this is the best model so far
        if val_metrics['masked_acc'] > best_val_masked_acc:
            best_val_masked_acc = val_metrics['masked_acc']
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
        
        # Save metrics to CSV for plotting later
        with open(os.path.join(log_dir, "training_metrics.csv"), "a") as f:
            f.write(f"{epoch+1},{current_lambda_mask:.4f},{current_lambda_fully_masked:.4f},{current_lambda_smoothness:.4f},"
                   f"{current_lambda_alignment:.4f},{train_metrics['dynamic_weight']:.4f},{train_metrics['total_loss']:.6f},{train_metrics['masked_loss']:.6f},"
                   f"{train_metrics['unmasked_loss']:.6f},{train_metrics['masking_loss']:.6f},"
                   f"{train_metrics['fully_masked_loss']:.6f},{train_metrics['smoothness_loss']:.6f},{train_metrics['alignment_loss']:.6f},"
                   f"{train_metrics['alignment_divergence']:.6f},{train_metrics['mask_mean']:.6f},"
                   f"{train_masked_pixels_pct:.2f},{train_metrics['binary_mask_mean']:.6f},"
                   f"{train_metrics['fully_masked_pct']:.2f},{train_metrics['masked_acc']:.2f},"
                   f"{train_metrics['unmasked_acc']:.2f},{val_metrics['total_loss']:.6f},"
                   f"{val_metrics['masked_loss']:.6f},{val_metrics['unmasked_loss']:.6f},"
                   f"{val_metrics['mask_mean']:.6f},{val_metrics['masked_pixels_pct']:.2f},"
                   f"{val_metrics['binary_mask_mean']:.6f},{val_metrics['fully_masked_pct']:.2f},"
                   f"{val_metrics['masked_acc']:.2f},{val_metrics['unmasked_acc']:.2f},"
                   f"{val_metrics['masking_loss']:.6f},{val_metrics['fully_masked_loss']:.6f},"
                   f"{val_metrics['smoothness_loss']:.6f},{val_metrics['alignment_loss']:.6f},"
                   f"{val_metrics['alignment_divergence']:.6f},{val_metrics['dynamic_weight']:.4f}\n")
        
        # Visualize results
        visualize_results(epoch, epoch_train_dir, epoch_val_dir, model, train_loader, val_loader, classes)
        
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(log_dir, f"model_epoch_{epoch+1}.pth"))

    
    
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, "log.eps"))
    writer.close()

    # print("Best accuracy:")
    # print(best_prec1)
    # Save final model
    torch.save(model.state_dict(), os.path.join(log_dir, "model_final.pth"))
    evaluate_model(test_loader, model, criterion)
    plot_training_metrics(log_dir)

def default_hyperparams(log_dir):
    # Hyperparameters
    batch_size = 256 
    num_epochs = 20  # Fewer epochs for quicker results
    resnet_learning_rate = 0.05 # initial learning rate
    unet_learning_rate = 0.001
    
    # U-Net weight freezing parameter
    start_epoch_unet = 1  # Epoch to start updating U-Net weights (0 = from beginning)

    # Masking loss hyperparameters
    initial_lambda_mask = 0.0 # Start with no masking penalty
    final_lambda_mask = 0.20  # End with strong masking penalty
    start_epoch_mask = 5  # Epoch to start applying mask loss (0 = from beginning)
    
    # Binary loss hyperparameters
    initial_lambda_fully_masked = 0.04  # Start with small fully masked loss weight
    final_lambda_fully_masked = 0.2 # End with stronger fully masked loss weight
    start_epoch_fully_masked = 2  # Epoch to start applying binary loss (0 = from beginning)
    
    # Smoothness regularization hyperparameters
    initial_lambda_smoothness = 0.01  # Start with some smoothness regularization
    final_lambda_smoothness = 0.01  # End with stronger smoothness regularization
    start_epoch_smoothness = 2  # Epoch to start applying smoothness loss (0 = from beginning)
    
    # Dynamic masked loss weighting parameters
    dynamic_masked_weight_min = 1.0  # Minimum weight multiplier
    dynamic_masked_weight_max = 2.0  # Maximum weight multiplier
    start_epoch_dynamic_weight = 1  # Epoch to start applying dynamic weighting (0 = from beginning)
    
    # Alignment loss hyperparameters
    initial_lambda_alignment = 0.5 # Start with moderate alignment loss weight
    final_lambda_alignment = 0.8    # End with stronger alignment loss weight
    start_epoch_alignment = 1  # Epoch to start applying alignment loss (0 = from beginning)
    
    # Radial mask hyperparameters
    radial_radius = 1  # Radius of influence for radial mask (in pixels)
    radial_decay = 0.4  # Decay factor for how quickly the influence decays with distance

    experiment_name = 'v7-fp32_celebA_resnet_50'

    # Print experiment configuration
    print(f"Experiment: {experiment_name}")
    print(f"Dataset: CelebA, LFWA+")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"ResNet Learning Rate: {resnet_learning_rate}")
    print(f"UNet Learning Rate: {unet_learning_rate}")
    print(f"U-Net Start Epoch: {start_epoch_unet} (weights frozen until this epoch)")
    print(f"Initial/Final Lambda Mask: {initial_lambda_mask}/{final_lambda_mask} (Start Epoch: {start_epoch_mask})")
    print(f"Initial/Final Lambda Fully Masked: {initial_lambda_fully_masked}/{final_lambda_fully_masked} (Start Epoch: {start_epoch_fully_masked})")
    print(f"Initial/Final Lambda Smoothness: {initial_lambda_smoothness}/{final_lambda_smoothness} (Start Epoch: {start_epoch_smoothness})")
    print(f"Initial/Final Lambda Alignment: {initial_lambda_alignment}/{final_lambda_alignment} (Start Epoch: {start_epoch_alignment})")
    print(f"Radial Mask Parameters: Radius={radial_radius}, Decay={radial_decay}")
    print(f"Dynamic Masked Weight Range:h {dynamic_masked_weight_min} to {dynamic_masked_weight_max} (Start Epoch: {start_epoch_dynamic_weight})")

    # Save hyperparameters as JSON
    hyperparameters = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "resnet_learning_rate": resnet_learning_rate,
        "unet_learning_rate": unet_learning_rate,
        "start_epoch_unet": start_epoch_unet,
        "initial_lambda_mask": initial_lambda_mask,
        "final_lambda_mask": final_lambda_mask,
        "start_epoch_mask": start_epoch_mask,
        "initial_lambda_fully_masked": initial_lambda_fully_masked,
        "final_lambda_fully_masked": final_lambda_fully_masked,
        "start_epoch_fully_masked": start_epoch_fully_masked,
        "initial_lambda_smoothness": initial_lambda_smoothness,
        "final_lambda_smoothness": final_lambda_smoothness,
        "start_epoch_smoothness": start_epoch_smoothness,
        "dynamic_masked_weight_min": dynamic_masked_weight_min,
        "dynamic_masked_weight_max": dynamic_masked_weight_max,
        "start_epoch_dynamic_weight": start_epoch_dynamic_weight,
        "initial_lambda_alignment": initial_lambda_alignment,
        "final_lambda_alignment": final_lambda_alignment,
        "start_epoch_alignment": start_epoch_alignment,
        "radial_radius": radial_radius,
        "radial_decay": radial_decay,
        "experiment_name": experiment_name,
    }
    
    # Save hyperparameters to JSON file
    with open(os.path.join(log_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)
    
    print(f"Hyperparameters saved to {os.path.join(log_dir, 'hyperparameters.json')}")
    return hyperparameters

# train one epoch
def train(train_loader, model, criterion, unet_optimizer, unet_params, resnet_optimizer, resnet_params, epoch, log_dir, train_metrics, hyperparams=None):

    
    # switch to train mode
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['num_epochs']}", mininterval=5)
    model.train()
    # unet_params      = list(model.mask_generator.parameters())
    # resnet_params  = [p for n, p in model.named_parameters()
    #                     if not n.startswith("mask_generator.")]

    masked_auroc = BinaryAUROC(num_tasks=40).to('cpu')
    unmasked_auroc = BinaryAUROC(num_tasks=40).to('cpu')
    use_amp = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for i, (input, target) in enumerate(train_loader_tqdm):
        input, target = input.to('cuda'), target.to('cuda')
        # measure data loading time
        # data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        unet_optimizer.zero_grad(set_to_none=True)  
        resnet_optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
        # compute output
            output = model(input)
            # if i % 100 == 0:
            #     print(output)
            # measure accuracy and record loss

            #  Losses #

            loss_dict = criterion(output, target)  # calcualte sum loss over all attributes.
            resnet_loss = loss_dict['resnet_loss']
            # Backward and optimize
            # optimizer.zero_grad()
            # loss_dict['total_loss'].backward()
            
            # optimizer.step()
            

            if epoch > hyperparams['start_epoch_unet']:
                unet_loss = loss_dict['unet_loss']
                total_loss = unet_loss + resnet_loss
            else: 
                total_loss = resnet_loss 
        # if i % 100 == 0:
        #     print('loss_dict')
        #     print(loss_dict)

        
        scaler.scale(total_loss).backward()
        scaler.step(resnet_optimizer)   
        if epoch > hyperparams['start_epoch_unet']:
            scaler.step(unet_optimizer)
        scaler.update()                  

        # Calculate accuracies
        masked_auroc.update(
            output['masked_logits'].T.detach().cpu(),   # (num_tasks, B) on CPU
            target.T.detach().cpu()                     # (num_tasks, B) on CPU
        )

        unmasked_auroc.update(
            output['unmasked_logits'].T.detach().cpu(),
            target.T.detach().cpu()
        )
        
        masked_acc, _ = get_accuracy(output['masked_logits'], target)
        unmasked_acc, _ = get_accuracy(output['unmasked_logits'], target)
        
        
        
        # Update metrics
        for key in ['total_loss', 'masked_loss', 'unmasked_loss', 'masking_loss', 
                    'fully_masked_loss', 'smoothness_loss', 'mask_mean', 
                    'binary_mask_mean', 'fully_masked_pct', 'dynamic_weight',
                    'alignment_loss', 'alignment_divergence']:  # Add new metrics
            if key in loss_dict:
                train_metrics[key] += loss_dict[key].item() if isinstance(loss_dict[key], torch.Tensor) else loss_dict[key]
        
        train_metrics['masked_acc'] += masked_acc
        train_metrics['unmasked_acc'] += unmasked_acc
        
        # Update tqdm description with current loss, binary mask metrics, dynamic weight, and alignment
        train_loader_tqdm.set_postfix(
            loss=f"{loss_dict['total_loss'].item():.4f}", 
            m_acc=f"{masked_acc:.2f}%",
            mask=f"{(1-loss_dict['mask_mean'].item())*100:.1f}%",
            masked=f"{loss_dict['fully_masked_pct']:.1f}%",
            visible=f"{100 * torch.mean((output['soft_mask'] > 0.8).float()).item():.1f}%",
            dw=f"{loss_dict['dynamic_weight'].item():.2f}",
            align=f"{loss_dict['alignment_divergence'].item():.2f}"  # Add alignment divergence
        )
    train_metrics['masked_binauroc'] = masked_auroc.compute()
    train_metrics['unmasked_binauroc'] = unmasked_auroc.compute()

    masked_auroc.reset()
    unmasked_auroc.reset()

    return train_metrics

# Helper function to get accuracy
def get_accuracy(logits, labels):
    """
    Compute accuracy per attribute and return the average.
    
    Args:
        logits (Tensor): shape (batch_size, num_attributes), raw outputs from the model
        labels (Tensor): shape (batch_size, num_attributes), ground truth binary labels (0 or 1)
    
    Returns:
        float: mean accuracy across all attributes
        list: individual accuracies for each attribute
    """
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(logits)

    # Convert probabilities to binary predictions
    preds = (probs >= 0.5).float()
    # print("labels")
    # print(labels)
    # print("preds")
    # print(preds)

    # Compare predictions to ground truth
    correct = (preds == labels.float()).float()
    # print("correct: ")
    # print(correct)

    # Accuracy per attribute (i.e., column-wise mean)
    per_attr_acc = correct.mean(dim=0)  # shape: (num_attributes,)

    # Mean accuracy across all attributes
    mean_acc = per_attr_acc.mean().item()

    

    return mean_acc, per_attr_acc.tolist()

# Function to visualize results from both train and val sets
def visualize_results(epoch, train_dir, val_dir, model, train_loader, val_loader, classes, num_samples=8):
    model.eval()
    
    # Visualize training samples
    print("visualizing samples from training data")
    visualize_dataset_samples(train_loader, epoch, num_samples, "train", model, classes, train_dir)
    
    print("visualizing samples from validation data")
    # Visualize validation samples
    visualize_dataset_samples(val_loader, epoch, num_samples, "val", model, classes, val_dir)

# Update the visualization function to show both soft and radial masks
def visualize_dataset_samples(data_loader, epoch, num_samples, dataset_type, model, classes, save_dir):
    # Get some examples
    dataiter = iter(data_loader)
    images, batch_labels = next(dataiter)
    
    # Select a subset of samples to visualize
    images = images[:num_samples].to('cuda')
    batch_labels = batch_labels[:num_samples].to('cuda')
    
    with torch.no_grad():
        outputs = model(images)
        soft_mask = outputs['soft_mask']
        radial_mask = outputs['radial_mask']
        binary_mask = outputs['binary_mask']  # Get binary mask
        masked_images = outputs['masked_input']
        
        # Get predictions

        masked_probs = torch.sigmoid(outputs['masked_logits'])
        unmasked_probs = torch.sigmoid(outputs['unmasked_logits'])

        # Convert probabilities to binary predictions
        masked_preds = (masked_probs >= 0.5).float()
        unmasked_preds = (unmasked_probs >= 0.5).float()
        # _, masked_preds = torch.max(outputs['masked_logits'], 1)
        # _, unmasked_preds = torch.max(outputs['unmasked_logits'], 1)
        
        # Move tensors to CPU for visualization
        images = images.cpu()
        soft_mask = soft_mask.cpu()
        radial_mask = radial_mask.cpu()
        binary_mask = binary_mask.cpu()  # Move binary mask to CPU
        masked_images = masked_images.cpu()
        batch_labels = batch_labels.cpu()
        masked_preds = masked_preds.cpu()
        unmasked_preds = unmasked_preds.cpu()
        
        # Create a figure with rows of images - now with 5 columns to show binary mask
        fig, axs = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
        
        for i in range(num_samples):
            # Original image with unmasked prediction
            img = images[i].numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5  # Unnormalize
            axs[i, 0].imshow(img)

            # Compute number of correct tags
            true_tags = (batch_labels[i] == 1)
            pred_tags = (unmasked_preds[i] == 1)
            num_correct_tags = (true_tags & pred_tags).sum().item()
            total_tags = true_tags.sum().item()
            
            # Display actual and predicted tags
            label_tags = [classes[j] for j in range(len(classes)) if batch_labels[i][j] == 1]
            pred_label_tags = [classes[j] for j in range(len(classes)) if unmasked_preds[i][j] == 1]

            axs[i, 0].set_title(f'Epoch {epoch}, Original:\n{", ".join(label_tags)}\nPred: {", ".join(pred_label_tags)}\nCorrect tags: {num_correct_tags}/{total_tags}')
            axs[i, 0].axis('off')

            # Soft Mask
            soft_mask_mean = soft_mask[i].mean().item()
            axs[i, 1].imshow(soft_mask[i].squeeze(), cmap='viridis')
            axs[i, 1].set_title(f'Soft Mask (1=Masked, 0=Visible)\nMean: {soft_mask_mean:.3f}')
            axs[i, 1].axis('off')

            # Radial Mask
            radial_mask_mean = radial_mask[i].mean().item()
            axs[i, 2].imshow(radial_mask[i].squeeze(), cmap='viridis')
            axs[i, 2].set_title(f'Radial Mask (1=Masked, 0=Visible)\nMean: {radial_mask_mean:.3f}')
            axs[i, 2].axis('off')

            # Binary Mask
            binary_mask_mean = binary_mask[i].mean().item()
            binary_pct = binary_mask_mean * 100
            axs[i, 3].imshow(binary_mask[i].squeeze(), cmap='binary')
            axs[i, 3].set_title(f'Binary Mask (> {model.upper_mask_level_threshold})\nMasked: {binary_pct:.1f}%')
            axs[i, 3].axis('off')

            # Masked image with prediction
            masked_img = masked_images[i].numpy().transpose((1, 2, 0))
            masked_img = masked_img * 0.5 + 0.5  # Unnormalize

            masked_pred_tags = [classes[j] for j in range(len(classes)) if masked_preds[i][j] == 1]
            masked_correct_tags = ((masked_preds[i] == 1) & (batch_labels[i] == 1)).sum().item()

            axs[i, 4].imshow(masked_img)
            axs[i, 4].set_title(f'Masked\nPred: {", ".join(masked_pred_tags)}\nCorrect tags: {masked_correct_tags}/{total_tags}')
            axs[i, 4].axis('off')

        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"results_samples.png"), dpi=150)
        plt.close(fig)
        
        # Save mask grids - soft, radial, and binary
        soft_mask_grid = torchvision.utils.make_grid(soft_mask.repeat(1, 3, 1, 1), nrow=4, normalize=True)
        torchvision.utils.save_image(soft_mask_grid, os.path.join(save_dir, f"soft_masks_grid.png"))
        
        radial_mask_grid = torchvision.utils.make_grid(radial_mask.repeat(1, 3, 1, 1), nrow=4, normalize=True)
        torchvision.utils.save_image(radial_mask_grid, os.path.join(save_dir, f"radial_masks_grid.png"))
        
        binary_mask_grid = torchvision.utils.make_grid(binary_mask.repeat(1, 3, 1, 1), nrow=4, normalize=True)
        torchvision.utils.save_image(binary_mask_grid, os.path.join(save_dir, f"binary_masks_grid.png"))
        
        # Save masked images grid
        masked_grid = torchvision.utils.make_grid(masked_images, nrow=4, normalize=True)
        torchvision.utils.save_image(masked_grid, os.path.join(save_dir, f"masked_images_grid.png"))

def plot_training_metrics(log_dir):
    import pandas as pd
    
    # Load metrics from CSV
    metrics_df = pd.read_csv(os.path.join(log_dir, "training_metrics.csv"))
    
    # Create plots
    plt.figure(figsize=(15, 35))  # Increased height for more plots
    
    # Losses plot - Training vs Validation
    plt.subplot(7, 2, 1)
    plt.plot(metrics_df['epoch'], metrics_df['train_masked_loss'], 'b-', label='Train Masked Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_masked_loss'], 'b--', label='Val Masked Loss')
    plt.plot(metrics_df['epoch'], metrics_df['train_unmasked_loss'], 'g-', label='Train Unmasked Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_unmasked_loss'], 'g--', label='Val Unmasked Loss')
    plt.title('Classification Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dynamic weight plot
    plt.subplot(7, 2, 2)
    plt.plot(metrics_df['epoch'], metrics_df['dynamic_weight'], 'r-', label='Train Dynamic Weight')
    plt.plot(metrics_df['epoch'], metrics_df['val_dynamic_weight'], 'r--', label='Val Dynamic Weight')
    plt.title('Dynamic Masked Loss Weight')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss gap vs Dynamic weight
    plt.subplot(7, 2, 3)
    # Calculate the loss gap
    metrics_df['train_loss_gap'] = metrics_df['train_masked_loss'] - metrics_df['train_unmasked_loss']
    plt.scatter(metrics_df['train_loss_gap'], metrics_df['dynamic_weight'], c=metrics_df['epoch'], cmap='viridis')
    plt.colorbar(label='Epoch')
    plt.title('Loss Gap vs Dynamic Weight')
    plt.xlabel('Masked-Unmasked Loss Gap')
    plt.ylabel('Dynamic Weight')
    plt.grid(True, alpha=0.3)
    
    # Masking loss plot
    plt.subplot(7, 2, 4)
    plt.plot(metrics_df['epoch'], metrics_df['train_masking_loss'], 'r-', label='Train Masking Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_masking_loss'], 'r--', label='Val Masking Loss')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_mask'], 'k--', label='Lambda Mask')
    plt.title('Masking Loss & Lambda')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Binary Masked loss plot
    plt.subplot(7, 2, 5)
    plt.plot(metrics_df['epoch'], metrics_df['train_fully_masked_loss'], 'c-', label='Train Binary Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_fully_masked_loss'], 'c--', label='Val Binary Loss')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_fully_masked'], 'k--', label='Lambda Binary')
    plt.title('Binary Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Smoothness loss plot
    plt.subplot(7, 2, 6)
    plt.plot(metrics_df['epoch'], metrics_df['train_smoothness_loss'], 'm-', label='Train Smoothness Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_smoothness_loss'], 'm--', label='Val Smoothness Loss')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_smoothness'], 'k--', label='Lambda Smoothness')
    plt.title('Smoothness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracies plot - Training vs Validation
    plt.subplot(7, 2, 7)
    plt.plot(metrics_df['epoch'], metrics_df['train_masked_acc'], 'b-', label='Train Masked Acc')
    plt.plot(metrics_df['epoch'], metrics_df['val_masked_acc'], 'b--', label='Val Masked Acc')
    plt.plot(metrics_df['epoch'], metrics_df['train_unmasked_acc'], 'g-', label='Train Unmasked Acc')
    plt.plot(metrics_df['epoch'], metrics_df['val_unmasked_acc'], 'g--', label='Val Unmasked Acc')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dynamic weight vs accuracies
    plt.subplot(7, 2, 8)
    plt.plot(metrics_df['dynamic_weight'], metrics_df['train_masked_acc'], 'bo-', label='Train Masked Acc')
    plt.plot(metrics_df['val_dynamic_weight'], metrics_df['val_masked_acc'], 'go-', label='Val Masked Acc')
    for i, txt in enumerate(metrics_df['epoch']):
        plt.annotate(txt, (metrics_df['dynamic_weight'].iloc[i], metrics_df['train_masked_acc'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
    plt.title('Dynamic Weight vs Masked Accuracy')
    plt.xlabel('Dynamic Weight')
    plt.ylabel('Masked Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Soft Masking percentage plot - Training vs Validation
    plt.subplot(7, 2, 9)
    plt.plot(metrics_df['epoch'], metrics_df['train_masked_pixels_pct'], 'm-', label='Train Soft Masked %')
    plt.plot(metrics_df['epoch'], metrics_df['val_masked_pixels_pct'], 'm--', label='Val Soft Masked %')
    plt.title('Soft Masking Percentage')
    plt.xlabel('Epoch')
    plt.ylabel('Masked Pixels (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Binary Masking percentage plot - Training vs Validation
    plt.subplot(7, 2, 10)
    plt.plot(metrics_df['epoch'], metrics_df['train_fully_masked_pct'], 'y-', label='Train Binary Masked %')
    plt.plot(metrics_df['epoch'], metrics_df['val_fully_masked_pct'], 'y--', label='Val Binary Masked %')
    plt.title('Binary Masking Percentage')
    plt.xlabel('Epoch')
    plt.ylabel('Masked Pixels (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Total loss plot
    plt.subplot(7, 2, 11)
    plt.plot(metrics_df['epoch'], metrics_df['train_total_loss'], 'b-', label='Train Total Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_total_loss'], 'b--', label='Val Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # All regularization parameters comparison (updated to include alignment)
    plt.subplot(7, 2, 12)
    plt.plot(metrics_df['epoch'], metrics_df['lambda_mask'], 'r-', label='Lambda Mask')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_fully_masked'], 'c-', label='Lambda Binary')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_smoothness'], 'm-', label='Lambda Smoothness')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_alignment'], 'y-', label='Lambda Alignment')
    plt.plot(metrics_df['epoch'], metrics_df['dynamic_weight'], 'g-', label='Dynamic Weight')
    plt.title('Regularization Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Alignment loss plot
    plt.subplot(7, 2, 13)
    plt.plot(metrics_df['epoch'], metrics_df['train_alignment_loss'], 'r-', label='Train Alignment Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_alignment_loss'], 'r--', label='Val Alignment Loss')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_alignment'], 'k--', label='Lambda Alignment')
    plt.title('Alignment Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Alignment divergence plot
    plt.subplot(7, 2, 14)
    plt.plot(metrics_df['epoch'], metrics_df['train_alignment_divergence'], 'b-', label='Train Alignment Divergence')
    plt.plot(metrics_df['epoch'], metrics_df['val_alignment_divergence'], 'b--', label='Val Alignment Divergence')
    plt.title('Feature Alignment Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Divergence (1-cosine_similarity)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_metrics.png"), dpi=150)
    plt.close()
    
    # Create a specific plot just for dynamic weight vs loss gap
    plt.figure(figsize=(10, 8))
    metrics_df['val_loss_gap'] = metrics_df['val_masked_loss'] - metrics_df['val_unmasked_loss']
    
    plt.scatter(metrics_df['train_loss_gap'], metrics_df['dynamic_weight'], 
                c=metrics_df['epoch'], cmap='viridis', s=100, alpha=0.7, label='Train')
    plt.scatter(metrics_df['val_loss_gap'], metrics_df['val_dynamic_weight'], 
                c=metrics_df['epoch'], cmap='viridis', s=100, marker='x', alpha=0.7, label='Validation')
    
    for i, txt in enumerate(metrics_df['epoch']):
        plt.annotate(txt, (metrics_df['train_loss_gap'].iloc[i], metrics_df['dynamic_weight'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
        plt.annotate(txt, (metrics_df['val_loss_gap'].iloc[i], metrics_df['val_dynamic_weight'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
    
    plt.colorbar(label='Epoch')
    plt.title('Loss Gap vs Dynamic Weight')
    plt.xlabel('Masked-Unmasked Loss Gap')
    plt.ylabel('Dynamic Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "dynamic_weight_analysis.png"), dpi=150)
    plt.close()
    
    # Create a new plot for alignment divergence vs. masked accuracy
    plt.figure(figsize=(10, 8))
    plt.scatter(metrics_df['train_alignment_divergence'], metrics_df['train_masked_acc'], 
                c=metrics_df['epoch'], cmap='viridis', s=100, alpha=0.7, label='Train')
    plt.scatter(metrics_df['val_alignment_divergence'], metrics_df['val_masked_acc'], 
                c=metrics_df['epoch'], cmap='viridis', s=100, marker='x', alpha=0.7, label='Validation')
    
    for i, txt in enumerate(metrics_df['epoch']):
        plt.annotate(txt, (metrics_df['train_alignment_divergence'].iloc[i], metrics_df['train_masked_acc'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
        plt.annotate(txt, (metrics_df['val_alignment_divergence'].iloc[i], metrics_df['val_masked_acc'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
    
    plt.colorbar(label='Epoch')
    plt.title('Alignment Divergence vs Masked Accuracy')
    plt.xlabel('Feature Alignment Divergence')
    plt.ylabel('Masked Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "alignment_analysis.png"), dpi=150)
    plt.close()

# Function to evaluate model on test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    
    print("EVALUATION ON TEST SET")
    
    test_metrics = validate_model(test_loader, model, criterion)
    
    # Print improved evaluation summary
    print(f"TEST RESULTS:")
    print(f"  Unmasked: Acc={test_metrics['unmasked_acc']:.2f}%, Loss={test_metrics['unmasked_loss']:.4f}")
    print(f"  Masked:   Acc={test_metrics['masked_acc']:.2f}%, Loss={test_metrics['masked_loss']:.4f}, Mask Loss={test_metrics['masking_loss']:.4f}, Fully Masked Loss={test_metrics['fully_masked_loss']:.4f}, Masked Pixels={test_metrics['masked_pixels_pct']:.2f}%")
    
    return test_metrics


def validate(val_loader, model, criterion):
    bar = Bar("Processing", max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    loss_avg = 0
    prec1_avg = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss

            #  ==== bceloss === #

            loss = criterion(output, target)  # calcualte sum loss over all attributes.
            losses.update(loss.item(), input.size(0))
            loss_avg = losses.avg

            top1.update(accuracy_bce(output, target).item(), input.size(0))
            prec1_avg = top1.avg

            # ================= #

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}".format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=loss_avg,
                top1=prec1_avg,
            )
            bar.next()
    bar.finish()
    return (loss_avg, prec1_avg)


def save_checkpoint(
    state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


# Update validate_model function to track dynamic weight
def validate_model(data_loader, model, criterion):
    model.eval()
    device = 'cuda'
    
    val_metrics = {
        'masked_tagwise_correct': 0,
        'unmasked_tagwise_correct': 0,
        'total_samples': 0,
        'total_tags': 0,
        'masked_loss': 0.0,
        'unmasked_loss': 0.0,
        'mask_mean': 0.0,
        'binary_mask_mean': 0.0,
        'fully_masked_pct': 0.0,
        'total_loss': 0.0,
        'masking_loss': 0.0,
        'fully_masked_loss': 0.0,
        'smoothness_loss': 0.0,
        'dynamic_weight': 0.0,
        'alignment_loss': 0.0,  # Add tracking for alignment loss
        'alignment_divergence': 0.0,  # Add tracking for raw divergence
        'masked_binauroc' : 0.0,
        'unmasked_binauroc' : 0.0
    }
            # 'mask_binauroc' : torch.zeros(40).to('cuda'),
        # 'unmask_binauroc' : torch.zeros(40).to('cuda')
    masked_auroc = BinaryAUROC(num_tasks=40).to(device)
    unmasked_auroc = BinaryAUROC(num_tasks=40).to(device)
    use_amp = not args.no_amp
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss_dict = criterion(outputs, labels)
                
            
            # _, masked_preds = torch.max(outputs['masked_logits'], 1)
            # _, unmasked_preds = torch.max(outputs['unmasked_logits'], 1)
                    # Get predictions

            masked_probs = torch.sigmoid(outputs['masked_logits'])
            unmasked_probs = torch.sigmoid(outputs['unmasked_logits'])

            # Convert probabilities to binary predictions
            masked_preds = (masked_probs >= 0.5).float()
            unmasked_preds = (unmasked_probs >= 0.5).float()


            batch_size = labels.size(0)
            val_metrics['total_samples'] += batch_size
            val_metrics['total_tags'] += labels.numel()
            val_metrics['masked_tagwise_correct'] += (masked_preds == labels).sum().item()
            val_metrics['unmasked_tagwise_correct'] += (unmasked_preds == labels).sum().item()
            
            unmasked_auroc.update(unmasked_probs.T, labels.T)
            masked_auroc.update(masked_probs.T, labels.T)

            # Sum losses
            val_metrics['total_loss'] += loss_dict['total_loss'].item() * batch_size
            val_metrics['masked_loss'] += loss_dict['masked_loss'].item() * batch_size
            val_metrics['unmasked_loss'] += loss_dict['unmasked_loss'].item() * batch_size
            val_metrics['masking_loss'] += loss_dict['masking_loss'].item() * batch_size
            val_metrics['fully_masked_loss'] += loss_dict['fully_masked_loss'].item() * batch_size
            val_metrics['smoothness_loss'] += loss_dict['smoothness_loss'].item() * batch_size
            val_metrics['dynamic_weight'] += loss_dict['dynamic_weight'].item() * batch_size
            val_metrics['alignment_loss'] += loss_dict['alignment_loss'].item() * batch_size  # Add alignment loss
            val_metrics['alignment_divergence'] += loss_dict['alignment_divergence'].item() * batch_size  # Add divergence
            
            # Sum mask means - both continuous and binary
            val_metrics['mask_mean'] += outputs['soft_mask'].mean().item() * batch_size
            val_metrics['binary_mask_mean'] += outputs['radial_mask'].mean().item() * batch_size
            val_metrics['fully_masked_pct'] += loss_dict['fully_masked_pct'] * batch_size
    
    # Calculate final metrics
    total = val_metrics['total_samples']
    
    return {
        'masked_acc': 100 * val_metrics['masked_tagwise_correct'] / val_metrics['total_tags'],
        'unmasked_acc': 100 * val_metrics['unmasked_tagwise_correct'] / val_metrics['total_tags'],
        'masked_loss': val_metrics['masked_loss'] / total,
        'unmasked_loss': val_metrics['unmasked_loss'] / total,
        'total_loss': val_metrics['total_loss'] / total,
        'mask_mean': val_metrics['mask_mean'] / total,
        'binary_mask_mean': val_metrics['binary_mask_mean'] / total,
        'masked_pixels_pct': (val_metrics['mask_mean'] / total) * 100,  # Use mask_mean directly (1 = masked)
        'fully_masked_pct': val_metrics['fully_masked_pct'] / total,
        'masking_loss': val_metrics['masking_loss'] / total,
        'fully_masked_loss': val_metrics['fully_masked_loss'] / total,
        'smoothness_loss': val_metrics['smoothness_loss'] / total,
        'dynamic_weight': val_metrics['dynamic_weight'] / total,
        'alignment_loss': val_metrics['alignment_loss'] / total,  # Add alignment loss average
        'alignment_divergence': val_metrics['alignment_divergence'] / total,  # Add divergence average
        'masked_binauroc' : masked_auroc.compute(),
        'unmasked_binauroc' : unmasked_auroc.compute(),
    }

def adjust_learning_rate(optimizer, epoch, init_lr):
    lr = optimizer.param_groups[0]["lr"]
    """Sets the learning rate to the initial LR decayed by 10 following schedule"""
    if args.lr_decay == "step":
        lr = init_lr * (args.gamma ** (epoch // args.step))
    elif args.lr_decay == "cos":
        lr =init_lr * (1 + cos(pi * epoch / args.epochs)) / 2
    elif args.lr_decay == "linear":
        lr = init_lr * (1 - epoch / args.epochs)
    elif args.lr_decay == "linear2exp":
        if epoch < args.turning_point + 1:
            # learning rate decay as 95% at the turning point (1 / 95% = 1.0526)
            lr = init_lr * (1 - epoch / int(args.turning_point * 1.0526))
        else:
            lr *= args.gamma
    elif args.lr_decay == "schedule":
        if epoch in args.schedule:
            lr *= args.gamma
    else:
        raise ValueError("Unknown lr mode {}".format(args.lr_decay))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == "__main__":
    main()
