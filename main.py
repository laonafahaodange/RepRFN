import argparse
from pprint import pprint
from dataloader import *
from trainer import *
from tester import *

import torch


def main(args):
    pprint(args)
    mode = args.mode
    if mode == 'train':
        train_loader = get_dataloader(mode='train', args=args)
        valid_loader = get_dataloader(mode='valid', args=args)

        # define your model here:
        from models.RepRFN import RepRFN
        model = RepRFN(upscale_factor=args.scale, deploy=False)

        if args.resume:
            assert args.checkpoint != '', 'Please Check The Param: arg.checkpoint'
            start_epoch = torch.load(args.checkpoint)['epoch']
            trainer = Trainer(args=args, train_loader=train_loader, valid_loader=valid_loader, my_model=model,
                              resume=True)
            for epoch in range(start_epoch, args.epochs):
                trainer.train(epoch)
                if epoch % args.val_per_epoch == 0:
                    trainer.valid(epoch)
                trainer.save(epoch)
        else:
            trainer = Trainer(args=args, train_loader=train_loader, valid_loader=valid_loader, my_model=model,
                              resume=False)
            for epoch in range(args.epochs):
                trainer.train(epoch)
                if epoch % args.val_per_epoch == 0:
                    trainer.valid(epoch)
                trainer.save(epoch)
    # TODO: 测试代码
    elif mode == 'test':
        test_loader = get_dataloader(mode='test', args=args)

        # define your model here:
        from models.RepRFN import RepRFN
        model = RepRFN(deploy=True, upscale_factor=args.scale)

        tester = Tester(args=args, test_loader=test_loader, my_model=model)
        tester.test()
    else:
        print('srcipt terminated, please check the param of "mode".')


if __name__ == '__main__':
    # 默认配置参数的设置
    parser = argparse.ArgumentParser("SISR SCRIPT ver.2023-06-05")

    # Mode specifications
    parser.add_argument('--mode', default='test', choices=('train', 'test'), help='Mode For (train | test)')

    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=8, help='Number Of Threads For Data Loading')
    parser.add_argument('--cpu', action='store_true', help='Use Cpu Only')
    parser.add_argument('--n_GPUs', type=int, default=1, help='Number Of GPUs')

    # Data specifications
    parser.add_argument("--train_hr_dir", default="D:/Datasets/SISR/DIV2K/DIV2K_train_HR_sub", type=str,
                        help="Train Dataset GT Image Directory")
    parser.add_argument("--train_lr_dir", default="D:/Datasets/SISR/DIV2K/DIV2K_train_LR_bicubic/X4_sub", type=str,
                        help="Train Dataset LR Image Directory")
    parser.add_argument("--valid_hr_dir", default="D:/Datasets/SISR/DIV2K/DIV2K_valid_HR", type=str,
                        help="Valid Dataset GT Image Directory")
    parser.add_argument("--valid_lr_dir", default="D:/Datasets/SISR/DIV2K/DIV2K_valid_LR_bicubic/X4", type=str,
                        help="Valid Dataset LR Image Directory")
    parser.add_argument("--test_hr_dir", default="D:/Datasets/SISR/Set5/GTmod12", type=str,
                        help="Test Dataset GT Image Directory")
    parser.add_argument("--test_lr_dir", default="D:/Datasets/SISR/Set5/LRbicx4", type=str,
                        help="Test Dataset LR Image Directory")
    parser.add_argument("--augment", default=True, action="store_true",
                        help="Data Augmentation, Includes hflip/vflip/rot90")

    # Model specifications
    parser.add_argument('--model', default='RepRFN', help='Model Name')
    parser.add_argument('--scale', type=int, default=4, help='Super Resolution Scale')
    parser.add_argument('--data_range', type=int, default=1, choices=(1, 255),
                        help='Train/Test Input Data Range')
    parser.add_argument('--data_format', type=str, default='rgb', choices=('bgr', 'rgb', 'ycbcr'),
                        help='Train/Test Input Data Format')
    parser.add_argument('--pre_train', type=str, default='D:/Project/Python/RepRFN/model_files/20230605/RepRFN_rep_epoch0_x4.pth',
                        help='Pre-trained Model Directory, Necessary When Test')
    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'),
                        help='FP Precision For Train And Test (single | half)')
    parser.add_argument("--rep", default=True, help="Whether The Model Uses Reparameters")
    parser.add_argument("--model_save_dir", default="./model_files/20230605", type=str,
                        help="Model Save Directory")
    parser.add_argument("--checkpoint_save_dir", default="./checkpoints/20230605", type=str,
                        help="Checkpoint Save Directory")

    # Training specifications
    parser.add_argument('--seed', type=int, default=42, help='Number Of Ramdon Seed')
    parser.add_argument("--batch_size", default=64, type=int, help="Batch Size")
    parser.add_argument("--epochs", default=1, type=int, help="Numbers of Epochs To Train")
    parser.add_argument('--loss_function', default='Custom', choices=('L1', 'L2', 'Charbonnier', 'Custom'),
                        help='Loss Fuction (L1 Loss | L2 loss | Charbonnier Loss)')
    parser.add_argument("--val_per_epoch", default=100, type=int, help="Valid PSNR/SSIM Every n Epochs")
    parser.add_argument("--log_dir", default="./logs/20230605", type=str,
                        help="Training LOG Save Directory")

    # Resume Training specifications
    parser.add_argument("--resume", default=False, help="Resume Training")

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--lr_scheduler', default='step',
                        choices=('step', 'multistep'),
                        help='LR Scheduler (StepLR | MultiStepLR)')
    parser.add_argument('--decay_step', type=int, default=100,
                        help='Learning Rate Decay Per Every N Step')
    parser.add_argument('--decay_milestone', type=str, default='200-400-600-800',
                        help='Learning Rate Decay In The epoch n1/n2/.../nn, In The Range Of epoch.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Learning Rate Decay Factor For Step Decay')
    parser.add_argument('--optimizer', default='ADAM',
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='Optimizer To Use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                        help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon For Numerical Stability, Default 1e-8, Try 1e-3 When The precision is "half"')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--gclip', type=float, default=0,
                        help='Gradient Clipping Threshold (0 = no clipping)')

    # Test specifications
    parser.add_argument('--self_ensemble', default=True, action='store_true', help='Use self-ensemble Method For Test')
    parser.add_argument("--psnr_ssim_y", default=True,
                        help="If True, Calculate PSNR/SSIM In Channel ycbcy_only_y, Else rgb")
    parser.add_argument("--test_result_dir", default="test_result/20230605/Set5/X4", type=str,
                        help="Test Result Save Dir")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
