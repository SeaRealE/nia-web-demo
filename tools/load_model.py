import sys
import os

now_dir = os.getcwd()
target_dir = '/model'
sys.path.append(now_dir + target_dir)


cp_seg_name = '/_checkpoint/cp_seg.pth'
cp_act_name = '/_checkpoint/cp_act.pth'
save_seg_name = '/model.pth'
save_act_name = '/action.pth'

# segmentation
import torch
from modeling.deeplab import DeepLab

n_class = 10

try:
    model = DeepLab(num_classes=n_class,
                    backbone='xception',
                    output_stride=16,
                    sync_bn=bool(None),
                    freeze_bn=bool(False))
    model = model.cuda()

    checkpoint = torch.load(now_dir + cp_seg_name)
    model.load_state_dict(checkpoint['state_dict'])
    torch.save(model, now_dir + target_dir + save_seg_name)
    print('segmentation model - OK!')
except:
    print('segmentation model - Failed!')

    
# action
import torchvision.models as models
import torch.nn as nn

try:
    model = models.resnet152()
    model.fc = nn.Sequential(nn.Linear(2048, 2048),
                     nn.BatchNorm1d(num_features=2048),
                     nn.ReLU(),
                     nn.Linear(2048, 1024),
                     nn.BatchNorm1d(num_features=1024),
                     nn.ReLU(),
                     nn.Linear(1024, 2))
    model = model.cuda()

    checkpoint = torch.load(now_dir + cp_act_name)
    model.load_state_dict(checkpoint)
    torch.save(model, now_dir + target_dir + save_act_name)
    print('action model - OK!')
except:
    print('action model - Failed!')
