from torch.utils import data
from utils.utils_image import *

"""Custom Dataset compatible with prebuilt DataLoader."""
"""
自定义读取数据集+增强
"""


class MyDataset(data.Dataset):
    def __init__(self, im_dir, gt_dir, data_range, data_format, mode='train', transform=None):
        self.im_paths = []
        self.gt_paths = []
        for fs in os.listdir(im_dir):
            self.im_paths.append(os.path.join(im_dir, fs))
        for fs in os.listdir(gt_dir):
            self.gt_paths.append(os.path.join(gt_dir, fs))
        self.im_paths.sort()
        self.gt_paths.sort()
        self.data_range = data_range
        self.data_format = data_format
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        im_path = self.im_paths[index]
        gt_path = self.gt_paths[index]
        # 训练时将数据缩放至0-1，计算loss
        if self.mode == 'train':
            # im = read_img(im_path)  # bgr or gray, hwc, 0-255 -> 0-1, ndarray
            # gt = read_img(gt_path)  # bgr or gray, hwc, 0-255 -> 0-1, ndarray
            im = read_img_v2(im_path, self.data_range, self.data_format, dtype=np.float)
            gt = read_img_v2(gt_path, self.data_range, self.data_format, dtype=np.float)
            if self.transform is not None:
                im, gt = self.transform(im, gt)
            im = single2tensor3(im)  # chw
            gt = single2tensor3(gt)  # chw
            return im, gt
        # 验证时将lr缩放至0-1输入网络，gt(即hr)不做处理
        elif self.mode == 'valid':
            # im = imread_uint(im_path)  # 0-255
            # gt = imread_uint(gt_path)  # 0-255
            im = read_img_v2(im_path, self.data_range, self.data_format, dtype=np.float)
            gt = read_img_v2(gt_path, 255, self.data_format, dtype=np.uint8)
            # im = uint2tensor3(im)  # 0-255 -> 0-1, float tensor
            im = single2tensor3(im)
            return im, gt
        elif self.mode == 'test':
            # im = imread_uint(im_path)  # 0-255
            # gt = imread_uint(gt_path)  # 0-255
            im = read_img_v2(im_path, self.data_range, self.data_format, dtype=np.float)
            gt = read_img_v2(gt_path, 255, self.data_format, dtype=np.uint8)
            # im = uint2tensor3(im)  # 0-255 -> 0-1, float tensor
            im = single2tensor3(im)
            gt_file_name = os.path.basename(gt_path).split('.')[0]
            # print(os.path.basename(gt_path))
            return im, gt, gt_file_name

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.im_paths)


def get_dataloader(mode, args):
    """Builds and returns Dataloader."""
    # transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = augment_img_v2
    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if mode == 'train':
        dataset = MyDataset(args.train_lr_dir, args.train_hr_dir, args.data_range, args.data_format, mode, transform)
        data_loader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=not args.cpu,
                                      num_workers=args.n_threads)
    elif mode == 'valid':
        dataset = MyDataset(args.valid_lr_dir, args.valid_hr_dir, args.data_range, args.data_format, mode, transform)
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, pin_memory=not args.cpu,
                                      num_workers=args.n_threads)
    elif mode == 'test':
        dataset = MyDataset(args.test_lr_dir, args.test_hr_dir, args.data_range, args.data_format, mode, transform)
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, pin_memory=not args.cpu,
                                      num_workers=args.n_threads)

    return data_loader
