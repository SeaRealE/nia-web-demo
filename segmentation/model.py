import numpy as np
import torch

from torchvision import transforms
from segmentation.dataloaders import custom_transforms as tr

from PIL import Image
import datetime

import fcn
import skimage.io

import cv2


label_list = ['11', '12', '13', '14', '15', '17'] 

color_list = [[128, 0, 0],  # 11 빨
             [0, 128, 0],   # 12 초
             [128, 128, 0], # 13 노
             [192, 0, 0],   # 14 진빨
             [0, 0, 128],   # 15 파
             [64, 0, 0]]    # 17 갈


def transform(sample):
    composed_transforms = transforms.Compose([
        tr.FixedResize(size=513),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()
    ])
    return composed_transforms(sample)

def untransform(img, lbl=None):
        mean_bgr = np.array([0.485, 0.456, 0.406])
        std_bgr = np.array([0.229, 0.224, 0.225])

        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= std_bgr
        img += mean_bgr
        img *= 255
        img = img.astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl


class MyModel:
    def __init__(self, pth):
        ## model load
        self.model = torch.load(pth)
        self.model = self.model.cuda()
        self.model.eval()

        ## image load
        ## src = Image.open('200311_Vid_1_000000.033.jpg').convert('RGB')
    def predict(self, src):
        src_width, src_height = src.size

        sample = {'image': src, 'label': None}     
        sample = transform(sample)
        image , target = torch.unsqueeze(sample['image'],0).cuda(), torch.unsqueeze(sample['label'],0).cuda()

        with torch.no_grad():
            output = self.model(image)
            
        imgs = image.data.cpu()[0]
        lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :][0]
        lbl_true = target.data.cpu() #cpu()

        imgs, lbl_true = untransform(imgs, lbl_true)
        viz = fcn.utils.visualize_segmentation(
            lbl_pred=lbl_pred, lbl_true=None, img=imgs, n_class=10,
            label_names = [' ',
                    '11_pforceps',
                    '12_mbforceps',
                    '13_mcscissors',
                    '15_pcapplier',
                    '18_pclip',
                    '20_sxir',
                    '19_mtclip',
                    '17_mtcapplier',
                    '14_graspers'])
        viz_no = fcn.utils.visualize_segmentation(lbl_pred=lbl_pred, lbl_true=None, img=imgs, n_class=10)

        w_min = int(viz.shape[1] / 3 * 1)
        w_max = int(viz.shape[1] / 3 * 2)



        result_img = viz_no[:,w_min:w_max,:].copy()
        result_img[result_img==127]=128
        result_img[result_img==63]=64
        result_img[result_img==191]=192

        mask_list = []
        ch_red, ch_green, ch_blue = result_img[:,:,0], result_img[:,:,1], result_img[:,:,2]
        for idx, color in enumerate(color_list):
            base = result_img.copy()

            r1, g1, b1 = color
            mask = (ch_red == r1) & (ch_green == g1) & (ch_blue == b1) 
            base[:,:,:3][np.invert(mask)] = [0, 0, 0]

            if not (base == np.zeros_like(base)).all():
                # crop mask
                mask_sum = base.sum(2)
                
    
                y, x = np.where(mask_sum > 0)
                h, w = mask_sum.shape

                y_max = y.max()
                y_min = y.min()
                x_max = x.max()
                x_min = x.min()

                y_max = h if y_max + 30 > h else y_max + 30
                y_min = 0 if y_min - 30 < 0 else y_min - 30
                x_max = w if x_max + 30 > w else x_max + 30
                x_min = 0 if x_min - 30 < 0 else x_min - 30

                base = imgs[y_min:y_max, x_min:x_max]

                mask_list.append([base, label_list[idx]])
                
                
                        
        result = Image.fromarray(np.uint8(viz[:,w_max:,:])).resize((src_width, src_height))
        
        return result, mask_list


