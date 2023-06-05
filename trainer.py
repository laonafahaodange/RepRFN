import copy
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.parallel as P
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.utils_image import *


# TODO: 需要增加断点继续训练的代码
class Trainer():
    def __init__(self, args, train_loader, valid_loader, my_model, resume):
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.args = args
        self.scale = args.scale
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.log_writer = SummaryWriter(args.log_dir)  # 保存训练数据日志的路径
        self.model = my_model.to(self.device)

        if args.optimizer == 'SGD':
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                       weight_decay=args.weight_decay)
        elif args.optimizer == 'ADAM':
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=args.lr, eps=args.epsilon)

        if args.lr_scheduler == 'step':
            self.scheduler = lrs.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)
        elif args.lr_scheduler == 'multistep':
            milestones = list(map(lambda x: int(x), args.decay.split('-')))
            self.scheduler = lrs.MultiStepLR(self.optimizer, milestones=milestones, gamma=args.gamma)

        if args.loss_function == 'L1':
            self.loss = nn.L1Loss()
        elif args.loss_function == 'L2':
            self.loss = nn.MSELoss()
        elif args.loss_function == 'Charbonnier':
            from losses.charbonnier_loss import CharbonnierLoss
            self.loss = CharbonnierLoss()
        # TODO: 自定义损失
        elif args.loss_function == 'Custom':
            from losses.custom_loss import CustomLoss
            self.loss = CustomLoss()

        if args.precision == 'half':
            self.model.half()

        if resume:
            checkpoint = torch.load(args.checkpoint)
            self.model.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_schedule'])
            self.optimizer.state_dict()['param_groups'][0]['lr'] = checkpoint['lr_last']
        else:
            if args.pre_train != '':
                self.model.load_state_dict(torch.load(args.pre_train))

    def train(self, epoch):
        lr_current = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.model.train()
        train_bar = tqdm(self.train_loader)
        loss_ls = []
        for batch, (lr_img, hr_img) in enumerate(train_bar):
            lr_img, hr_img = self.prepare(lr_img, hr_img)
            self.optimizer.zero_grad()
            if self.args.n_GPUs > 1:
                sr_img = P.data_parallel(self.model, lr_img, range(self.args.n_GPUs))
            else:
                sr_img = self.model(lr_img)
            loss = self.loss(sr_img, hr_img)
            loss_ls.append(loss)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            train_bar.desc = f'train epoch:[{epoch}], loss:{loss}'  # 可视化训练进程
        self.scheduler.step()
        ave_loss = sum(loss_ls) / len(loss_ls)
        self.log_writer.add_scalar('learning rate', lr_current, epoch)
        self.log_writer.add_scalar('average loss', ave_loss, epoch)

    def valid(self, epoch):
        torch.set_grad_enabled(False)
        self.model.eval()
        valid_bar = tqdm(self.valid_loader)
        psnr_ls = []
        ssim_ls = []
        for batch, (lr_img, hr_img) in enumerate(valid_bar):
            lr_img, _ = self.prepare(lr_img, hr_img)
            sr_img = self.model(lr_img)
            sr_img = tensor2uint(sr_img, self.args.data_range)
            hr_img = hr_img.squeeze()  # 1HWC -> HWC
            # valid 会在modcrop将hr_img转移到cpu
            hr_img = modcrop(hr_img, self.args.scale)
            border = self.args.scale
            psnr = calculate_psnr(sr_img, hr_img, border=border)
            ssim = calculate_ssim(sr_img, hr_img, border=border)
            psnr_ls.append(psnr)
            ssim_ls.append(ssim)
            valid_bar.desc = f'psnr:{psnr:.5f},ssim:{ssim:.5f}'  # 可视化训练进程
        torch.set_grad_enabled(True)
        ave_psnr = sum(psnr_ls) / len(psnr_ls)
        ave_ssim = sum(ssim_ls) / len(ssim_ls)
        self.log_writer.add_scalar('average psnr', ave_psnr, epoch)
        self.log_writer.add_scalar('average ssim', ave_ssim, epoch)

    def save(self, epoch):
        cwd = os.getcwd()
        model_save_path = os.path.join(cwd, self.args.model_save_dir)
        mkdir(model_save_path)
        if self.args.rep is True:
            # 保存重参数化后的模型
            model = copy.deepcopy(self.model)
            for module in model.modules():
                if hasattr(module, 'switch_to_deploy'):
                    module.switch_to_deploy()
            model_file_path = os.path.join(model_save_path,
                                           f'{self.args.model}_rep_epoch{epoch}_x{self.args.scale}.pth')
            torch.save(model.state_dict(), model_file_path)
        model_file_path = os.path.join(model_save_path, f'{self.args.model}_epoch{epoch}_x{self.args.scale}.pth')
        torch.save(self.model.state_dict(), model_file_path)
        # 保存断点训练所需的权重
        checkpoint = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "lr_schedule": self.scheduler.state_dict(),
            "lr_last": self.optimizer.state_dict()['param_groups'][0]['lr']
        }
        checkpoint_save_path = os.path.join(cwd, self.args.checkpoint_save_dir)
        mkdir(checkpoint_save_path)
        checkpoint_file_path = os.path.join(checkpoint_save_path, f'checkpoint_last.pth')
        torch.save(checkpoint, checkpoint_file_path)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(self.device)

        return [_prepare(a) for a in args]
