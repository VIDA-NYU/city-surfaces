"""
Copyright 2020 Nvidia Corporation
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import random
import torch
from runx.logx import logx
from citysurfaces.config import assert_and_infer_cfg, cfg
from citysurfaces.utils.misc import AverageMeter, prep_experiment, eval_metrics
from citysurfaces.utils.misc import ImageDumper
from citysurfaces.utils.trnval_utils import eval_minibatch, validate_topn
from citysurfaces.loss.utils import get_loss
from citysurfaces.loss.optimizer import restore_net


import citysurfaces.datasets as datasets
import citysurfaces.network as network
import pandas as pd
import numpy as np
import copy
import datetime
from citysurfaces.resources import Project

#from tile2net.Data import Data

# Autoresume is disabled by default to avoid untrusted path injection.
AutoResume = None


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--arch', type=str, default='ocrnet.HRNet_Mscale',
                    help='Network architecture')
parser.add_argument('--dataset', type=str, default='citysurfaces',
                    help='name of your dataset')

parser.add_argument('--num_workers', type=int, default=4,
                    help='cpu worker threads per dataloader instance')
parser.add_argument('--cv', type=int, default=0,
                    help=('Cross-validation split id to use. Default # of splits set'
                          ' to 3 in config'))

parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=512,
                    help='tile size for class uniform sampling')


parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by DDP')
parser.add_argument('--global_rank', default=0, type=int,
                    help='parameter used by DDP')

parser.add_argument('--bs_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--snapshot', type=str, default=None, help = 'path to the model weight')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--result_dir', type=str, default=None,
                    help='where to write log output')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump augmented images for sanity checks')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--dump_separate', action='store_true', default=False,
                    help='Dump input/gt/pred images')
parser.add_argument('--dump_overlay', action='store_true', default=False,
                    help='Dump composited overlay images')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help=('Minimum testing to verify nothing failed, '
                          'Runs code for 1 epoch of train and val'))
parser.add_argument('--multi_scale_inference', action='store_true',
                    help='Run multi scale inference')

parser.add_argument('--default_scale', type=float, default=1.0,
                    help='default scale to run validation')

parser.add_argument('--log_msinf_to_tb', action='store_true', default=False,
                    help='Log multi-scale Inference to Tensorboard')

parser.add_argument('--eval', type=str, default=None,
                    help=('just run evaluation, can be set to val or trn or '
                          'folder'))
parser.add_argument('--eval_folder', type=str, default=None,
                    help='path to frames to evaluate')

parser.add_argument('--three_scale', action='store_true', default=False)
parser.add_argument('--alt_two_scale', action='store_true', default=False)
parser.add_argument('--do_flip', action='store_true', default=False)
parser.add_argument('--extra_scales', type=str, default='0.5,1.5,2.0')
parser.add_argument('--n_scales', type=str, default=None)
parser.add_argument('--align_corners', action='store_true',
                    default=False)
parser.add_argument('--translate_aug_fix', action='store_true', default=False)
parser.add_argument('--mscale_lo_scale', type=float, default=0.5,
                    help='low resolution training scale')
parser.add_argument('--pre_size', type=int, default=None,
                    help=('resize long edge of images to this before'
                          ' augmentation'))
parser.add_argument('--amp_opt_level', default='O1', type=str,
                    help=('amp optimization level'))
parser.add_argument('--dump_topn', type=int, default=50,
                    help='Dump worst val images')

parser.add_argument('--dump_assets', action='store_true',
                    help='Dump interesting assets')
parser.add_argument('--dump_all_images', action='store_true',
                    help='Dump all images, not just a subset')
parser.add_argument('--dump_for_submission', action='store_true',
                    help='Dump assets for submission')
parser.add_argument('--dump_topn_all', action='store_true', default=True,
                    help='dump topN worst failures')


parser.add_argument('--deterministic', action='store_true',
                    default=False, help='Set CUDNN Deterministic')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Enable torch.cuda.amp autocast for eval')
parser.add_argument('--unsafe_load', action='store_true', default=False,
                    help='Allow weights_only=False when loading snapshots')
parser.add_argument('--download_weights', action='store_true', default=False,
                    help='Download pretrained weights to assets/seg_weights')
parser.add_argument('--weights_url', type=str,
                    default='https://drive.google.com/drive/folders/1cu-MATHgekWUYqj9TFr12utl6VB-XKSu',
                    help='Weights folder URL')
parser.add_argument('--summary', action='store_true',
                    default=False)
parser.add_argument('--segattn_bot_ch', type=int, default=None,
                    help='bottleneck channels for seg and attn heads')
parser.add_argument('--grad_ckpt', action='store_true',
                    default=False)
parser.add_argument('--distributed', type=int, default=False,)
parser.add_argument('--no_metrics', action='store_true', default=False,
                    help='prevent calculation of metrics')
parser.add_argument('--supervised_mscale_loss_wt', type=float, default=None,
                    help='weighting for the supervised loss')
parser.add_argument('--ocr_aux_loss_rmi', action='store_true', default=False,
                    help='allow rmi for aux loss')




args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

torch.backends.cudnn.deterministic = args.deterministic
torch.backends.cudnn.benchmark = not args.deterministic

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2


num_gpus = torch.cuda.device_count()

if num_gpus > 1:
    # Distributed training setup

    args.world_size = int(os.environ.get('WORLD_SIZE', num_gpus))
    dist.init_process_group(backend='nccl', init_method='env://')
    args.local_rank = dist.get_rank()
    torch.cuda.set_device(args.local_rank)
    args.distributed = True
    print(f'Using distributed training with {args.world_size} GPUs.')
elif num_gpus == 1:
    # Single GPU setup
    print('Using a single GPU.')
    args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
else:
    # CPU setup
    print('Using CPU.')
    args.local_rank = -1  # Indicating CPU usage




def check_termination(epoch):
    if AutoResume:
        shouldterminate = AutoResume.termination_requested()
        if shouldterminate:
            if args.global_rank == 0:
                progress = "Progress %d%% (epoch %d of %d)" % (
                    (epoch * 100 / args.max_epoch),
                    epoch,
                    args.max_epoch
                )
                AutoResume.request_resume(
                    user_dict={"RESUME_FILE": logx.save_ckpt_fn,
                               "TENSORBOARD_DIR": args.result_dir,
                               "EPOCH": str(epoch)
                               }, message=progress)
                return 1
            else:
                return 1
    return 0


def main():
    """
    Main Function
    """
    if AutoResume:
        AutoResume.init()

    set_random_seed(args.seed, args.deterministic)

    assert args.result_dir is not None, 'need to define result_dir arg'
    logx.initialize(logdir=args.result_dir,
                    tensorboard=True, hparams=vars(args),
                    global_rank=args.global_rank)

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn
    assert_and_infer_cfg(args)
    prep_experiment(args)
    _, val_loader, _ = datasets.setup_loaders(args)
    criterion, criterion_val = get_loss(args)

    if args.download_weights:
        weights_dir = Project.resources.assets.weights.path
        expected = [
            Project.resources.assets.weights.block_c_10classes.path,
            Project.resources.assets.weights.hrnetv2_w48_imagenet_pretrained.path,
        ]
        if not all(p.exists() for p in expected):
            Project.resources.assets.weights.download(args.weights_url)


    if args.snapshot:
        if 'ASSETS_PATH' in args.snapshot:
            args.snapshot = args.snapshot.replace('ASSETS_PATH', cfg.ASSETS_PATH)
        if not os.path.isfile(args.snapshot):
            raise FileNotFoundError(f"Snapshot not found: {args.snapshot}")
        try:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            checkpoint = torch.load(args.snapshot, map_location=torch.device('cpu'), weights_only=True)
        except Exception as exc:
            if args.unsafe_load:
                logx.msg(f"weights_only load failed ({exc}); retrying with weights_only=False")
                checkpoint = torch.load(args.snapshot, map_location=torch.device('cpu'), weights_only=False)
            else:
                raise
        args.restore_net = True
        msg = f"Loading weights from: checkpoint={args.snapshot}"
        logx.msg(msg)
    else:
        logx.msg("No snapshot provided; running with random weights.")


    net = network.get_net(args, criterion)
    net = network.wrap_network_in_dataparallel(args, net)
    if args.snapshot:
        restore_net(net, checkpoint)

    if args.summary:

        from thop import profile
        img = torch.randn(1, 3, 640, 640).cuda()
        mask = torch.randn(1, 1, 640, 640).cuda()
        macs, params = profile(net, inputs={'images': img, 'gts': mask})
        print(f'macs {macs} params {params}')
        sys.exit()


    torch.cuda.empty_cache()

    # Evaluation modes:
    #  --eval val         : run validation (with optional dump_assets/dump_topn)
    #  --eval folder      : run on a folder without metrics
    #  --eval test        : smoke test on val_loader

    if args.eval == 'test':
        validate(val_loader, net, criterion=None, epoch=0,
                 calc_metrics=False, dump_assets=args.dump_assets,
                 dump_all_images=True, testing=True)
        return 0

    if args.eval == 'val':
        if args.dump_topn:
            validate_topn(val_loader, net, criterion_val, 0, args)
        else:
            validate(val_loader, net, criterion=criterion_val, epoch=0,
                     dump_assets=args.dump_assets,
                     dump_all_images=args.dump_all_images,
                     calc_metrics=not args.no_metrics)
        return 0

    if args.eval == 'folder':
        validate(val_loader, net, criterion=criterion_val, epoch=0,
                 calc_metrics=False, dump_assets=args.dump_assets,
                 dump_all_images=True, testing=True)
        return 0

    if args.eval is not None:
        raise ValueError(f"unknown eval option {args.eval}")

    

def validate(val_loader, net, criterion, epoch,
             calc_metrics=True,
             dump_assets=False,
             dump_all_images=False, testing=None):
    """
    Run validation for one epoch
    :val_loader: data loader for validation
    :net: the network
    :criterion: loss fn
    :epoch: current epoch
    :calc_metrics: calculate validation score
    :dump_assets: dump attention prediction(s) images
    :dump_all_images: dump all images, not just N
    """
    dumper = ImageDumper(val_len=len(val_loader),
                         dump_all_images=dump_all_images,
                         dump_assets=dump_assets,
                         dump_separate=args.dump_separate,
                         dump_overlay=args.dump_overlay,
                         dump_for_submission=args.dump_for_submission)

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    pred = dict()
    _temp = dict.fromkeys([i for i in range(10)], None)
    for val_idx, data in enumerate(val_loader):
        input_images, labels, img_names, _ = data
        if args.dump_for_submission:
            submit_fn = '{}.png'.format(img_names[0])
            if val_idx % 20 == 0:
                logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')
            if os.path.exists(os.path.join(dumper.save_dir, submit_fn)):
                continue

        # Run network
        assets, _iou_acc = \
            eval_minibatch(data, net, criterion, val_loss, calc_metrics,
                           args, val_idx)

        iou_acc += _iou_acc

        input_images, labels, img_names, _ = data

        if testing:
            prediction = assets['predictions'][0]
            values, counts = np.unique(prediction, return_counts=True)
            pred[img_names[0]]=copy.copy(_temp)
            for v in range(len(values)):
                pred[img_names[0]][values[v]]=counts[v]

            # pred.update({img_names[0]: dict(zip(values, counts))})
            dumper.dump({'gt_images': labels,'input_images': input_images,'img_names': img_names, 'assets': assets},
                        val_idx, testing=True)
        else:
            dumper.dump({'gt_images': labels,
                         'input_images': input_images,
                         'img_names': img_names,
                         'assets': assets}, val_idx)

        if val_idx > 5 and args.test_mode:
            break

        if val_idx % 20 == 0:
            logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')

    was_best = False
    if calc_metrics:
        was_best = eval_metrics(iou_acc, args, net, None, val_loss, epoch)

    # Write out a summary html page and tensorboard image table
    if not args.dump_for_submission:
        dumper.write_summaries(was_best)

    if testing:
        sufix = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
        df = pd.DataFrame.from_dict(pred, orient='index')
        df.to_csv(os.path.join(dumper.save_dir,f'raw_numb_{sufix}.csv'), mode = 'a+')
        df_p = df.div(df.sum(axis=1), axis=0)
        # df_p.to_csv(os.path.join(dumper.save_dir,'freq.csv'), mode = 'a+', header=False)
        df_p.to_csv(os.path.join(dumper.save_dir,f'freq_{sufix}.csv'), mode = 'a+')



if __name__ == '__main__':
    main()
