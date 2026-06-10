import os.path as osp
import torch
import torch.utils.data as data
import basicsr.data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os
import functools

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _image_file_list(root):
    return natsorted([
        path for path in glob.glob(osp.join(root, '*'))
        if osp.isfile(path) and path.lower().endswith(IMG_EXTENSIONS)
    ])


def _read_rgb_tensor(path, size=None):
    img = util.read_img(None, path, size)
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(
        np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img


def _paired_random_crop_tensor(img_lq, img_gt, gt_size):
    _, h, w = img_gt.shape
    pad_h = max(0, gt_size - h)
    pad_w = max(0, gt_size - w)
    if pad_h != 0 or pad_w != 0:
        pad = (0, pad_w, 0, pad_h)
        img_lq = F.pad(img_lq.unsqueeze(0), pad, mode='replicate').squeeze(0)
        img_gt = F.pad(img_gt.unsqueeze(0), pad, mode='replicate').squeeze(0)
        _, h, w = img_gt.shape

    top = random.randint(0, h - gt_size)
    left = random.randint(0, w - gt_size)
    img_lq = img_lq[:, top:top + gt_size, left:left + gt_size]
    img_gt = img_gt[:, top:top + gt_size, left:left + gt_size]
    return img_lq, img_gt


class Dataset_SDSDEventImage(data.Dataset):
    """RGB frame pairs from the SDSD event release.

    Expected layout:
        dataroot/
            pair1/
                low/*.png
                normal/*.png
                low_event/*.h5 or event npz files are ignored

    Low and normal frame timestamps can differ, so frames are paired by
    natural sorted order inside each scene.
    """

    def __init__(self, opt):
        super(Dataset_SDSDEventImage, self).__init__()
        self.opt = opt
        self.root = osp.expanduser(opt['dataroot'])
        self.low_dir = opt.get('low_dir', 'low')
        self.gt_dir = opt.get('gt_dir', 'normal')
        self.train_size = opt.get('train_size', None)
        self.gt_size = opt.get('gt_size', None)
        self.use_flip = opt.get('use_flip', opt.get('geometric_augs', True))
        self.use_rot = opt.get('use_rot', opt.get('geometric_augs', True))

        if not osp.isdir(self.root):
            raise FileNotFoundError(f'SDSD dataroot does not exist: {self.root}')

        self.paths = self._scan_pairs()
        if len(self.paths) == 0:
            raise ValueError(f'No SDSD RGB image pairs found in {self.root}')

    def _scan_pairs(self):
        pairs = []
        scene_dirs = [
            path for path in glob.glob(osp.join(self.root, '*'))
            if osp.isdir(path)
        ]
        for scene_dir in natsorted(scene_dirs):
            scene = osp.basename(scene_dir)
            low_folder = osp.join(scene_dir, self.low_dir)
            gt_folder = osp.join(scene_dir, self.gt_dir)
            if not osp.isdir(low_folder) or not osp.isdir(gt_folder):
                continue

            low_paths = _image_file_list(low_folder)
            gt_paths = _image_file_list(gt_folder)
            if len(low_paths) != len(gt_paths):
                raise ValueError(
                    f'SDSD pair count mismatch in {scene}: '
                    f'{len(low_paths)} low images vs {len(gt_paths)} GT images')

            for frame_idx, (lq_path, gt_path) in enumerate(zip(low_paths, gt_paths)):
                pairs.append({
                    'lq_path': lq_path,
                    'gt_path': gt_path,
                    'scene': scene,
                    'frame_idx': frame_idx
                })
        return pairs

    def __getitem__(self, index):
        index = index % len(self.paths)
        pair = self.paths[index]

        img_lq = _read_rgb_tensor(pair['lq_path'], self.train_size)
        img_gt = _read_rgb_tensor(pair['gt_path'], self.train_size)

        if self.opt['phase'] == 'train':
            if self.gt_size is not None:
                img_lq, img_gt = _paired_random_crop_tensor(
                    img_lq, img_gt, self.gt_size)
            img_lq, img_gt = util.augment_torch(
                [img_lq, img_gt], self.use_flip, self.use_rot)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'folder': pair['scene'],
            'idx': f"{pair['frame_idx']}/{len(self.paths)}",
            'border': 0,
            'scene': pair['scene'],
            'frame_idx': pair['frame_idx'],
            'lq_path': pair['lq_path'],
            'gt_path': pair['gt_path']
        }

    def __len__(self):
        return len(self.paths)


class Dataset_SDSDImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_SDSDImage, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.io_backend_opt = opt['io_backend']
        self.data_type = self.io_backend_opt['type']
        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}

        if opt['testing_dir'] is not None:
            testing_dir = opt['testing_dir']
            testing_dir = testing_dir.split(',')
        else:
            testing_dir = []
        print('testing_dir', testing_dir)

        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)

        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            # for frames in each video:
            subfolder_name = osp.basename(subfolder_GT)

            if self.opt['phase'] == 'train':
                if (subfolder_name in testing_dir):
                    continue

                if (subfolder_name.split('_2')[0] in testing_dir):
                    continue
            else:  # val test
                if not(subfolder_name in testing_dir) and not(subfolder_name.split('_2')[0] in testing_dir):
                    continue

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)

            max_idx = len(img_paths_LQ)
            assert max_idx == len(
                img_paths_GT), 'Different number of images in LQ and GT folders'
            self.data_info['path_LQ'].extend(
                img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)

            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

            border_l = [0] * max_idx
            for i in range(self.half_N_frames):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            if self.cache_data:
                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        img_LQ_path = self.imgs_LQ[folder][idx:idx + 1]
        img_GT_path = self.imgs_GT[folder][idx:idx + 1]

        img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
        img_LQ = img_LQ[0]
        img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
        img_GT = img_GT[0]

        if self.opt['phase'] == 'train':

            # LQ_size = self.opt['LQ_size']
            # GT_size = self.opt['GT_size']

            # _, H, W = img_GT.shape  # real img size

            # rnd_h = random.randint(0, max(0, H - GT_size))
            # rnd_w = random.randint(0, max(0, W - GT_size))
            # img_LQ = img_LQ[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            # img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            rlt = util.augment_torch(
                img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]

        # img_nf = img_LQ.clone().permute(1, 2, 0).numpy() * 255.0
        # img_nf = cv2.blur(img_nf, (5, 5))
        # img_nf = img_nf * 1.0 / 255.0
        # img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)

        return {
            'lq': img_LQ,
            'gt': img_GT,
            # 'nf': img_nf,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': img_LQ_path[0],
            'gt_path': img_GT_path[0]
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])
