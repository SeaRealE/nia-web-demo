import sys
import os

now_dir = os.getcwd()
target_dir = '/segmentation'
cp_name = '/_checkpoint/cp_seg.pth'
save_name = '/model.pth'

sys.path.append(now_dir + target_dir)


import torch
from modeling.deeplab import DeepLab

n_class = 10

model = DeepLab(num_classes=n_class,
                backbone='xception',
                output_stride=16,
                sync_bn=bool(None),
                freeze_bn=bool(False))
model = model.cuda()

checkpoint = torch.load(now_dir + cp_name)
model.load_state_dict(checkpoint['state_dict'])
torch.save(model, now_dir + target_dir + save_name)