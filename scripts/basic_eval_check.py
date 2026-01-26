import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from runx.logx import logx
from citysurfaces.val import validate
from citysurfaces.config import update_dataset_cfg, update_dataset_inst


class DummyDataset(Dataset):
    def __init__(self, n=2, h=64, w=64, num_classes=4):
        self.n = n
        self.h = h
        self.w = w
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.rand(3, self.h, self.w)
        mask = torch.randint(0, self.num_classes, (self.h, self.w), dtype=torch.uint8)
        return img, mask, f"dummy_{idx}", 1.0


class DummyNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs['images']
        pred = self.conv(x)
        return {'pred': pred}


class DummyDatasetInst:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.id_to_trainid = {i: i for i in range(num_classes)}
        self.trainid_to_name = {i: str(i) for i in range(num_classes)}
        self.color_mapping = [0] * (256 * 3)

    def colorize_mask(self, mask):
        return Image.fromarray(mask.astype('uint8'))


if __name__ == '__main__':
    logx.initialize(logdir="/tmp/citysurfaces_basic_eval", tensorboard=False)

    num_classes = 4
    update_dataset_cfg(num_classes, ignore_label=-1)
    update_dataset_inst(DummyDatasetInst(num_classes))

    dataset = DummyDataset(n=2, h=64, w=64, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    net = DummyNet(num_classes).cuda()

    validate(loader, net, criterion=None, epoch=0,
             calc_metrics=False, dump_assets=False, dump_all_images=False)
    print("basic eval check ok")
