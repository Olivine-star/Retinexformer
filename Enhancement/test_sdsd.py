# Retinexformer SDSD inference.

import argparse
import os
from os import path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from tqdm import tqdm

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
ENHANCEMENT_DIR = osp.abspath(osp.dirname(__file__))
if ENHANCEMENT_DIR not in sys.path:
    sys.path.insert(0, ENHANCEMENT_DIR)

import utils
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils.options import parse


def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x

    outputs = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rotate in [False, True]:
                outputs.append(forward_transformed(x, hflip, vflip, rotate))
    return torch.stack(outputs).mean(dim=0)


def load_checkpoint(model, weights, strict=True):
    if not osp.isfile(weights):
        raise FileNotFoundError(f'Checkpoint does not exist: {weights}')

    checkpoint = torch.load(weights, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = {
        k[7:] if k.startswith('module.') else k: v
        for k, v in state_dict.items()
    }
    bare_model = model.module if hasattr(model, 'module') else model
    bare_model.load_state_dict(state_dict, strict=strict)


def main():
    parser = argparse.ArgumentParser(
        description='SDSD low-light image enhancement using Retinexformer')
    parser.add_argument('--opt', required=True, type=str, help='Path to option YAML file.')
    parser.add_argument('--weights', required=True, type=str, help='Path to model weights.')
    parser.add_argument('--output_dir', default='', type=str, help='Directory for enhanced images.')
    parser.add_argument('--gpus', default='0', type=str, help='GPU devices.')
    parser.add_argument('--cpu', action='store_true', help='Run inference on CPU.')
    parser.add_argument('--self_ensemble', action='store_true', help='Use x8 self-ensemble.')
    parser.add_argument('--non_strict_load', action='store_true', help='Allow missing or unexpected checkpoint keys.')
    parser.add_argument('--max_images', default=0, type=int, help='Only infer the first N images when > 0.')
    args = parser.parse_args()

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    if args.cpu or not torch.cuda.is_available():
        opt['num_gpu'] = 0
    device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')

    model = create_model(opt).net_g
    load_checkpoint(model, args.weights, strict=not args.non_strict_load)
    model.eval()

    val_opt = opt['datasets']['val']
    dataset = create_dataset(val_opt)
    dataloader = create_dataloader(dataset, val_opt, num_gpu=opt['num_gpu'], dist=False)

    config = osp.splitext(osp.basename(args.opt))[0]
    checkpoint = osp.splitext(osp.basename(args.weights))[0]
    output_dir = args.output_dir or osp.join('results', 'SDSD', config, checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    psnr_values = []
    ssim_values = []
    factor = int(opt.get('val', {}).get('window_size', 4)) or 4

    with torch.inference_mode():
        for idx, data_batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            if args.max_images > 0 and idx >= args.max_images:
                break

            input_ = data_batch['lq'].to(device)
            target = data_batch['gt'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            lq_path = data_batch['lq_path'][0]
            scene = data_batch.get('scene', data_batch.get('folder', ['default']))[0]

            _, _, h, w = input_.shape
            pad_h = (factor - h % factor) % factor
            pad_w = (factor - w % factor) % factor
            if pad_h != 0 or pad_w != 0:
                input_ = F.pad(input_, (0, pad_w, 0, pad_h), 'reflect')

            if args.self_ensemble:
                restored = self_ensemble(input_, model)
            else:
                restored = model(input_)

            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach()
            restored = restored.permute(0, 2, 3, 1).squeeze(0).numpy()

            scene_dir = osp.join(output_dir, scene)
            os.makedirs(scene_dir, exist_ok=True)
            save_path = osp.join(scene_dir, osp.splitext(osp.basename(lq_path))[0] + '.png')
            utils.save_img(save_path, img_as_ubyte(restored))

            psnr_values.append(utils.PSNR(target, restored))
            ssim_values.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))

    print(f'Enhanced images saved to: {output_dir}')
    if psnr_values:
        print('PSNR: %.6f' % np.mean(np.array(psnr_values)))
        print('SSIM: %.6f' % np.mean(np.array(ssim_values)))


if __name__ == '__main__':
    main()
