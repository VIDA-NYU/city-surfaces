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

Dataset setup and loaders
"""

import importlib
import torchvision.transforms as standard_transforms

from citysurfaces.transforms import joint_transforms
from citysurfaces.transforms import transforms as extended_transforms
from torch.utils.data import DataLoader

from citysurfaces.config import cfg, update_dataset_cfg, update_dataset_inst
from runx.logx import logx
from citysurfaces.datasets.randaugment import RandAugment


def setup_loaders(args):
    """
    Setup Data Loaders
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    # TODO add error checking to make sure class exists
    logx.msg(f'dataset = {args.dataset}')

    mod = importlib.import_module('citysurfaces.datasets.{}'.format(args.dataset))
    dataset_cls = getattr(mod, 'Loader')

    logx.msg(f'ignore_label = {dataset_cls.ignore_label}')

    update_dataset_cfg(num_classes=dataset_cls.num_classes,
                       ignore_label=dataset_cls.ignore_label)

    ######################################################################
    # Define transformations, augmentations
    ######################################################################

    # Joint transformations (eval-only: no training augmentations)
    train_joint_transform_list = []

    ######################################################################
    # Image only augmentations
    ######################################################################
    train_input_transform = None

    mean_std = (cfg.DATASET.MEAN, cfg.DATASET.STD)

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()


    target_train_transform = None

    val_joint_transform_list = None

    if args.eval is None or args.eval == 'val':
        val_name = 'val'
    elif args.eval == 'folder':
        val_name = 'folder'
    elif args.eval == 'test':
        val_name = 'test'
    else:
        raise ValueError('unknown eval mode {}'.format(args.eval))

    ######################################################################
    # Create loaders
    ######################################################################
    val_set = dataset_cls(
        mode=val_name,
        joint_transform_list=val_joint_transform_list,
        img_transform=val_input_transform,
        label_transform=target_transform,
        eval_folder=args.eval_folder)

    update_dataset_inst(dataset_inst=val_set)

    if args.distributed:
        from citysurfaces.datasets.sampler import DistributedSampler
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False,
                                         consecutive_sample=False)
    else:
        val_sampler = None

    val_loader = DataLoader(val_set, batch_size=args.bs_val,
                            num_workers=args.num_workers // 2,
                            shuffle=False, drop_last=False,
                            sampler=val_sampler)

    train_set = None
    train_loader = None

    return train_loader, val_loader, train_set
