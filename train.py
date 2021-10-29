import os
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import DiceLoss
import cv2
import logging
from glob import glob
from torch.utils.data import Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

backbone = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 
'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 
'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 
'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 
'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 
'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception', 
'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 
'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 
'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 
'timm-tf_efficientnet_lite4', 'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 
'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 
'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 
'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 
'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 
'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 
'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 
'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 
'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 'timm-mobilenetv3_large_075', 
'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075', 
'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100', 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l']

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def second2time(second):
    if second < 60:
        return str('{}'.format(round(second, 4)))
    elif second < 60*60:
        m = second//60
        s = second % 60
        return str('{}:{}'.format(int(m), round(s, 1)))
    elif second < 60*60*60:
        h = second//(60*60)
        m = second % (60*60)//60
        s = second % (60*60) % 60
        return str('{}:{}:{}'.format(int(h), int(m), int(s)))

def inial_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

class SegDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')
        image = cv2.imread(img_file[0], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        return {
            'image': image,
            'label': mask.long(),
        }
    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            img_trans = img_nd.transpose(2, 0, 1)
        except:
            print(img_nd.shape)
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans


train_transform = A.Compose([
    A.Resize(3072, 6144),
    A.VerticalFlip(p=0.3),
    A.HorizontalFlip(p=0.3),        
    A.Affine(p=0.3),
    A.ColorJitter(p=0.2),
    A.OneOf([
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
    ], p=0.5),
    A.ElasticTransform(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(3072, 6144),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

train_imgs_dir = 'train_images/'
val_imgs_dir = 'val_images/'

train_labels_dir = 'train_labels/'
val_labels_dir = 'val_labels/'


train_data = SegDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
valid_data = SegDataset(val_imgs_dir, val_labels_dir, transform=val_transform)

model = smp.UnetPlusPlus(encoder_name='timm-tf_efficientnet_lite0', encoder_weights="imagenet", classes=2, decoder_channels=(192, 96, 48, 24, 12)).cuda()


ckpt_dir = './ckpt/'
log_dir = './log/'


if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def train(model, train_data, valid_data):
    # 初始化
    model_name = 'timm-tf_efficientnet_lite0'
    epochs = 200
    batch_size = 1
    lr = 0.009
    weight_decay = 5e-4
    iter_inter = 50
    save_log_dir = log_dir
    save_ckpt_dir = ckpt_dir
    device = 'cuda'

    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.AdamW(model.parameters(), lr ,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6, last_epoch=-1)
    criterion = DiceLoss(mode='multiclass').cuda()
    logger = inial_logger(os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) +'_'+model_name+ '.log'))

    
    epoch_lr = []
    train_loader_size = train_loader.__len__()
    best_dice = 0
    epoch_start = 0

    logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size, valid_data_size))

    for epoch in range(epoch_start, epochs):
        epoch_start = time.time()
        #训练
        model.train()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target = batch_samples['image'], batch_samples['label']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            pred = model(data)
            loss = criterion(pred, target) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / train_loader_size)
            train_iter_loss.update(loss.item)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg,spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()

        #验证
        model.eval()
        dc_list = []
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target = batch_samples['image'], batch_samples['label']
                data, target = Variable(data.to(device)), Variable(target.to(device))
                pred = model(data)
                loss = criterion(pred, target)
                pred = pred.cpu().data.numpy()
                pred = np.argmax(pred,axis=1)
                pd = copy.deepcopy(pred)
                target = target.cpu().data.numpy()
                tgt = copy.deepcopy(target)
                pred = pred.flatten()
                target = target.flatten()
                inter = np.sum(pred * target)
                union = np.sum(pred) + np.sum(target)
                dc = (2*inter + 0.001)/ (union + 0.001)
                dc_list.append(dc)
        

        mean_dice = sum(dc_list) / len(dc_list)
        print(mean_dice)
        logger.info('[val] epoch:{} mean dice:{:.2f}'.format(epoch, mean_dice))                


        epoch_lr.append(optimizer.param_groups[0]['lr'])

        if mean_dice > best_dice:
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best-dice{:.5f}-epoch{}.pth'.format(mean_dice, epoch))
            torch.save(model.state_dict(), filename)
            best_dice = mean_dice
            logger.info('[save] Best Dice Model saved at epoch:{} ============================='.format(epoch))
            


train(model, train_data, valid_data)
