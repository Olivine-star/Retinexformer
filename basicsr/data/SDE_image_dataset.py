import glob
import os
from os import path as osp

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted


IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _image_paths(folder):
    paths = [
        v for v in glob.glob(osp.join(folder, '*'))
        if osp.isfile(v) and v.lower().endswith(IMG_EXTENSIONS)
    ]
    return natsorted(paths)


class Dataset_SDEImage(data.Dataset):
    """Paired RGB image dataset for SDE low-light enhancement.

    Expected split layout:
        dataroot/
            scene_0/
                low/*.png
                normal/*.png
            scene_1/
                low/*.png
                normal/*.png

    SDE frame timestamps differ between low and normal folders, so frames are
    paired by natural sorted order inside each scene.
    """

    def __init__(self, opt):
        super(Dataset_SDEImage, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.low_dir = opt.get('low_dir', 'low')
        self.gt_dir = opt.get('gt_dir', 'normal')
        self.root = osp.expanduser(opt['dataroot'])

        if not osp.isdir(self.root):
            raise FileNotFoundError(f'SDE dataroot does not exist: {self.root}')

        self.paths = self._scan_pairs()
        if len(self.paths) == 0:
            raise ValueError(f'No SDE image pairs found in {self.root}')

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', True)

    def _scan_pairs(self):
        pairs = []
        scene_dirs = [
            v for v in glob.glob(osp.join(self.root, '*')) if osp.isdir(v)
        ]
        for scene_dir in natsorted(scene_dirs):
            scene = osp.basename(scene_dir)
            low_folder = osp.join(scene_dir, self.low_dir)
            gt_folder = osp.join(scene_dir, self.gt_dir)
            if not osp.isdir(low_folder) or not osp.isdir(gt_folder):
                continue

            low_paths = _image_paths(low_folder)
            gt_paths = _image_paths(gt_folder)
            if len(low_paths) != len(gt_paths):
                raise ValueError(
                    f'SDE pair count mismatch in {scene}: '
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
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        pair = self.paths[index]

        gt_path = pair['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception as exc:
            raise RuntimeError(f'GT path {gt_path} is not readable') from exc

        lq_path = pair['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception as exc:
            raise RuntimeError(f'LQ path {lq_path} is not readable') from exc

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_lq, img_gt = padding(img_lq, img_gt, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'scene': pair['scene'],
            'frame_idx': pair['frame_idx']
        }

    def __len__(self):
        return len(self.paths)
