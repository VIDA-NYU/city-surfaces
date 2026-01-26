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

Generic dataloader base class
"""
import os
import glob
import json
import numpy as np
import torch

from PIL import Image
from torch.utils import data
from citysurfaces.config import cfg
from runx.logx import logx
from citysurfaces.utils.misc import tensor_to_pil


class BaseLoader(data.Dataset):
    def __init__(self, quality, mode, joint_transform_list, img_transform,
                 label_transform):

        super(BaseLoader, self).__init__()
        self.quality = quality
        self.mode = mode
        self.joint_transform_list = joint_transform_list
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.train = mode == 'train'
        self.id_to_trainid = {}
        self.centroids = None
        self.all_imgs = None
        self.drop_mask = np.zeros((1024, 2048))
        self.drop_mask[15:840, 14:2030] = 1.0
        self.read_failures = []
        # Optional path to write newline-delimited JSON records of read failures.
        self.failed_reads_report = os.environ.get('FAILED_READS_REPORT')

    def build_epoch(self):
        """Set current epoch images (uniform sampling disabled for inference)."""
        self.imgs = self.all_imgs

    @staticmethod
    def find_images(img_root, mask_root, img_ext, mask_ext):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.
        """
        img_exts = (img_ext,) if isinstance(img_ext, str) else tuple(img_ext)
        mask_exts = (mask_ext,) if isinstance(mask_ext, str) else tuple(mask_ext)

        items = []
        for ext in img_exts:
            img_path = '{}/*.{}'.format(img_root, ext)
            imgs = glob.glob(img_path)
            for full_img_fn in imgs:
                _, img_fn = os.path.split(full_img_fn)
                img_name, _ = os.path.splitext(img_fn)

                matching_mask = None
                for m_ext in mask_exts:
                    full_mask_fn = os.path.join(mask_root, f'{img_name}.{m_ext}')
                    if os.path.exists(full_mask_fn):
                        matching_mask = full_mask_fn
                        break

                if matching_mask is None:
                    raise FileNotFoundError(f"Mask for {full_img_fn} not found in {mask_root} (extensions tried: {mask_exts})")
                items.append((full_img_fn, matching_mask))

        if not items:
            raise FileNotFoundError(f"No images found in {img_root} with extensions {img_exts}")
        return sorted(items)

    def disable_coarse(self):
        pass

    def colorize_mask(self, image_array):
        """
        Colorize the segmentation mask
        """
        new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.color_mapping)
        return new_mask

    def dump_images(self, img_name, mask, centroid, class_id, img):
        img = tensor_to_pil(img)
        outdir = 'new_dump_imgs_{}'.format(self.mode)
        os.makedirs(outdir, exist_ok=True)
        if centroid is not None:
            dump_img_name = '{}_{}'.format(self.trainid_to_name[class_id],
                                           img_name)
        else:
            dump_img_name = img_name
        out_img_fn = os.path.join(outdir, dump_img_name + '.png')
        out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
        out_raw_fn = os.path.join(outdir, dump_img_name + '_mask_raw.png')
        mask_img = self.colorize_mask(np.array(mask))
        raw_img = Image.fromarray(np.array(mask))
        img.save(out_img_fn)
        mask_img.save(out_msk_fn)
        raw_img.save(out_raw_fn)

    def do_transforms(self, img, mask, centroid, img_name, class_id):
        """
        Do transformations to image and mask

        :returns: image, mask
        """
        scale_float = 1.0

        if img.size != mask.size:
            msg = f"Image/mask size mismatch for {img_name}: img{img.size} vs mask{mask.size}"
            logx.msg(msg)
            raise ValueError(msg)

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK! Assume the first transform accepts a centroid
                    outputs = xform(img, mask, centroid)
                else:
                    outputs = xform(img, mask)

                if len(outputs) == 3:
                    img, mask, scale_float = outputs
                else:
                    img, mask = outputs

        if self.img_transform is not None:
            img = self.img_transform(img)

        if cfg.DATASET.DUMP_IMAGES:
            self.dump_images(img_name, mask, centroid, class_id, img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)

        return img, mask, scale_float

    def read_images(self, img_path, mask_path, mask_out=False):
        allow_fallback = os.environ.get("ALLOW_BLACK_FALLBACK", "").lower() in ("1", "true", "yes")
        try:
            img = Image.open(img_path).convert('RGB')
            if mask_path is None or mask_path == '':
                w, h = img.size
                mask = np.zeros((h, w))
            else:
                mask = Image.open(mask_path)
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"Failed to read {img_path} or {mask_path}: {exc}") from exc
            blck = np.zeros((512, 512))
            img = Image.fromarray(np.uint8(blck)).convert('RGB')
            mask = Image.fromarray(np.uint8(blck))
            self._record_read_failure(img_path, mask_path, exc)
            logx.msg(f"{img_path} or {mask_path} unreadable ({exc!r}); replaced with black placeholder.")


        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        if mask_out:
            mask = self.drop_mask * mask
        mask = mask.copy()
        for k, v in self.id_to_trainid.items():
            binary_mask = (mask == k) #+ (gtCoarse == k)
            mask[binary_mask] = v

        mask = Image.fromarray(mask.astype(np.uint8))
        return img, mask, img_name


    def __getitem__(self, index):
        """
        Generate data:

        :return:
        - image: image, tensor
        - mask: mask, tensor
        - image_name: basename of file, string
        """
        # Pick an image, fill in defaults if not using class uniform
        if len(self.imgs[index]) == 2:
            img_path, mask_path = self.imgs[index]
            centroid = None
            class_id = None
        else:
            img_path, mask_path, centroid, class_id = self.imgs[index]

        img, mask, img_name = self.read_images(img_path, mask_path,
                                               mask_out=False)


        if 'refinement' in mask_path:
            
            mask = np.array(mask)
            prob_mask_path = mask_path.replace('.png', '_prob.png')
            # put it in 0 to 1
            prob_map = np.array(Image.open(prob_mask_path)) / 255.0
            prob_map_threshold = (prob_map < cfg.DATASET.CUSTOM_COARSE_PROB)
            mask[prob_map_threshold] = cfg.DATASET.IGNORE_LABEL
            mask = Image.fromarray(mask.astype(np.uint8))

        img, mask, scale_float = self.do_transforms(img, mask, centroid,
                                                    img_name, class_id)

        return img, mask, img_name, scale_float

    def __len__(self):
        return len(self.imgs)

    def calculate_weights(self):
        raise BaseException("not supported yet")

    def _record_read_failure(self, img_path, mask_path, exc):
        entry = {"image": img_path, "mask": mask_path, "error": repr(exc)}
        self.read_failures.append(entry)
        report_path = getattr(self, "failed_reads_report", None)
        if not report_path:
            return
        dir_name = os.path.dirname(report_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        try:
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as log_exc:
            logx.msg(f"Failed to write failure report to {report_path}: {log_exc}")
