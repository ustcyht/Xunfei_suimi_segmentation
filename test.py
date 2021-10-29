# -*- coding: utf-8 -*-
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import glob
import os
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transform = A.Compose([
        A.Resize(3072, 6144),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
def img_resize(out_path):
    img = np.array(Image.open(out_path))
    img = img*255
    img = resize(img, (500, 1024))
    img = img*255
    img[img < 150] = 0
    img[img >= 150] = 255
    img = img.astype(np.uint8)
    img = Image.fromarray(np.uint8(img)).convert('L')
    img.save(out_path)


def test(img_dir):
    image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(image=image)['image']
    img = img.unsqueeze(0)
    with torch.no_grad():
        img=img.cuda()
        output = model(img)
    pred = output.squeeze().cpu().data.numpy()
    pred = np.argmax(pred,axis=0)
    return pred

if __name__=="__main__":

    model = smp.UnetPlusPlus(encoder_name='timm-tf_efficientnet_lite0', classes=2, decoder_channels=(192, 96, 48, 24, 12)).cuda()
    pth = 'ckpt/checkpoint-best-dice0.28219115136980855-epoch0.pth' # 权重文件路径
    model.load_state_dict(torch.load(pth))
    model.eval()
    out_dir='result/' # 输出路径
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    test_paths = glob.glob('test/images/*')# 测试集路径
    for per_path in tqdm(test_paths):
        result = test(per_path)
        img = Image.fromarray(np.uint8(result))
        img = img.convert('L')
        out_path = os.path.join(out_dir, per_path.split('/')[-1][:-4]+'.png')
        img.save(out_path) 
        img_resize(out_path)

        
                



