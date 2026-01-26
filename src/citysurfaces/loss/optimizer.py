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

# Optimizer and scheduler related tasks

import math
import os
import torch

from torch import optim
from runx.logx import logx

from citysurfaces.config import cfg
from citysurfaces.loss.radam import RAdam


def _split_param_groups(param_iterable, weight_decay):
    """
    Create parameter groups splitting decay / no-decay (norm/bias) params.
    """
    decay, no_decay = [], []
    for name, param in param_iterable:
        if not param.requires_grad:
            continue
        if name.endswith('.bias') or 'bn' in name.lower() or 'norm' in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    if not decay and not no_decay:
        return [{'params': [p for _, p in param_iterable], 'weight_decay': weight_decay}]
    groups = []
    if decay:
        groups.append({'params': decay, 'weight_decay': weight_decay})
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0})
    return groups


def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    if args.max_epoch <= 0:
        raise ValueError("max_epoch must be > 0")
    if args.poly_exp <= 0:
        raise ValueError("poly_exp must be > 0")

    named_params = list(net.named_parameters())
    if getattr(args, 'separate_wd', False):
        param_groups = _split_param_groups(named_params, args.weight_decay)
    else:
        param_groups = [p for _, p in named_params]

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(param_groups,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=False)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(param_groups,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=args.amsgrad)
    elif args.optimizer == 'radam':
        optimizer = RAdam(param_groups,
                          lr=args.lr,
                          weight_decay=args.weight_decay)
    else:
        raise ValueError('Not a valid optimizer')

    def poly_schd(epoch):
        return math.pow(max(1 - epoch / args.max_epoch, 0.0), args.poly_exp)

    def poly2_schd(epoch):
        if epoch < args.poly_step:
            poly_exp = args.poly_exp
        else:
            poly_exp = 2 * args.poly_exp
        return math.pow(max(1 - epoch / args.max_epoch, 0.0), poly_exp)

    warmup_epochs = max(getattr(args, 'warmup_epochs', 0), 0)
    warmup_factor = max(getattr(args, 'warmup_factor', 0.0), 0.0)

    def apply_warmup(base_lambda):
        if warmup_epochs == 0:
            return base_lambda
        def wrapper(epoch):
            if epoch < warmup_epochs:
                warm_ratio = warmup_factor + (1 - warmup_factor) * epoch / float(max(warmup_epochs, 1))
                return warm_ratio
            return base_lambda(epoch)
        return wrapper

    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_EPOCH == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_EPOCH
        scale_value = args.rescale
        lambda1 = lambda epoch: \
             math.pow(max(1 - epoch / args.max_epoch, 0.0),
                      args.poly_exp) if epoch < rescale_thresh else scale_value * math.pow(
                          max(1 - (epoch - rescale_thresh) / (args.max_epoch - rescale_thresh), 0.0),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=apply_warmup(lambda1))
    elif args.lr_schedule == 'poly2':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=apply_warmup(poly2_schd))
    elif args.lr_schedule == 'poly':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=apply_warmup(poly_schd))
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler


def apply_gradient_clipping(optimizer, max_norm, norm_type=2.0):
    """
    Clip gradients for all parameter groups.
    """
    if max_norm is None:
        return
    for group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'],
                                       max_norm=max_norm,
                                       norm_type=norm_type)


def load_weights(net, optimizer, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    if not os.path.isfile(snapshot_file):
        raise FileNotFoundError(f"Checkpoint not found: {snapshot_file}")
    logx.msg("Loading weights from model {}".format(snapshot_file))
    net, optimizer = restore_snapshot(net, optimizer, snapshot_file, restore_optimizer_bool)
    return net, optimizer


def restore_snapshot(net, optimizer, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    try:
        checkpoint = torch.load(snapshot, map_location=torch.device('cpu'), weights_only=True)
    except TypeError:
        # PyTorch <2.1 does not support weights_only; fallback with a loud warning.
        logx.msg("weights_only not supported by this torch version, loading full checkpoint (less safe)")
        checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logx.msg("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer


def restore_opt(optimizer, checkpoint):
    assert 'optimizer' in checkpoint, 'cant find optimizer in checkpoint'
    optimizer.load_state_dict(checkpoint['optimizer'])


def restore_net(net, checkpoint):
    assert 'state_dict' in checkpoint, 'cant find state_dict in checkpoint'
    forgiving_state_restore(net, checkpoint['state_dict'])


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """

    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        new_k = k
        if new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size():
            new_loaded_dict[k] = loaded_dict[new_k]
        else:            
            logx.msg("Skipped loading parameter {}".format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net
