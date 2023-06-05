from tqdm import tqdm
import basicsr.metrics.psnr_ssim

from utils.utils_image import *


# TODO: 增加测试任意数据集代码
class Tester():
    def __init__(self, args, test_loader, my_model):
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.args = args
        self.scale = args.scale
        self.test_loader = test_loader
        self.model = my_model.to(self.device)

        assert args.pre_train != '', 'Please Check The Param: arg.pretrain'
        self.model.load_state_dict(torch.load(args.pre_train))

    def test(self):
        cwd = os.getcwd()
        test_result_path = os.path.join(cwd, self.args.test_result_dir)
        mkdir(test_result_path)
        torch.set_grad_enabled(False)
        self.model.eval()
        test_bar = tqdm(self.test_loader)
        psnr_ls = []
        ssim_ls = []
        for batch, (lr_img, hr_img, hr_file_name) in enumerate(test_bar):
            lr_img, _ = self.prepare(lr_img, hr_img)
            # sr_img,out1,out2,out3,out4 = self.model(lr_img)
            if self.args.self_ensemble:
                sr_img=forward_x8(lr_img,self.args,self.model.forward)
            else:
                sr_img = self.model(lr_img)
            sr_img = tensor2uint(sr_img, self.args.data_range)  # 0-1 -> 0-255, uint8

            sr_img_for_save = sr_img.copy()
            # test 需要先对hr转移到cpu上才能继续往下
            hr_img = hr_img.data.squeeze().float().cpu().numpy().astype(np.uint8)

            # if self.args.psnr_ssim_y:
            #     sr_img = rgb2ycbcr(sr_img, only_y=True)
            #     hr_img = rgb2ycbcr(hr_img, only_y=True)
            # hr_img = modcrop(hr_img, self.args.scale)
            border = self.args.scale
            # psnr = calculate_psnr(sr_img, hr_img, border=border)
            # # calculate_ssim的输入必须是uint8
            # ssim = calculate_ssim(sr_img, hr_img, border=border)
            # import basicsr.metrics.psnr_ssim
            psnr = basicsr.metrics.psnr_ssim.calculate_psnr(hr_img, sr_img, crop_border=border, test_y_channel=self.args.psnr_ssim_y)
            ssim = basicsr.metrics.psnr_ssim.calculate_ssim(hr_img, sr_img, crop_border=border, test_y_channel=self.args.psnr_ssim_y)

            psnr_ls.append(psnr)
            ssim_ls.append(ssim)
            test_result_img_path = os.path.join(test_result_path, f'{hr_file_name[0]}_x{self.args.scale}.png')
            imsave_v2(sr_img_for_save, test_result_img_path, self.args.data_format)
            # test_bar.desc = f'psnr:{psnr:.5f},ssim:{ssim:.5f},save in {test_result_img_path}'
        torch.set_grad_enabled(True)
        ave_psnr = sum(psnr_ls) / len(psnr_ls)
        ave_ssim = sum(ssim_ls) / len(ssim_ls)
        print(f'ave_psnr:{ave_psnr}, ave_ssim:{ave_ssim}')
        print(f'{ave_psnr:.2f}/{ave_ssim:.4f}')

    def prepare(self, *args):
        def _prepare(tensor):
            return tensor.to(self.device)

        return [_prepare(a) for a in args]
