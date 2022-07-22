from dataset.dataset_falre import SemiDataset
from model import archs
from pinggu import get_free_gpu_number,get_used_cpu

import argparse

import os
import time
import threading
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import SimpleITK as sitk
import numpy as np
import nrrd

MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='FLARE2022')

    # basic settings
    parser.add_argument('--data-root', type=str, default='./dataset/val')
    parser.add_argument('--dataset', type=str, default='flare')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--num-class', type=int, default=14)



    parser.add_argument('--model', type=str,
                        default='Unext')


    parser.add_argument('--pred-mask-path', type=str,default='./pred_mask')

    parser.add_argument('--save-path', type=str, default='./save_model')

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str,default='./dataset/splits')
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')

    args = parser.parse_args()
    return args

def parse_args2():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=24, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext')

    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=14, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')

    # loss


    # dataset
    parser.add_argument('--dataset', default='FLARE22',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.nrrd',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.nrrd',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')


    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=2, type=int)

    config = parser.parse_args()

    return config

def init_basic_elems(args):
    # create model
    config = vars(parse_args2())
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels']
                                           )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load('./save_model/Unext_0.74.pth',map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('./save_model/Unext_0.74.pth'))
    #model.load_state_dict(torch.load('./save_model/Unext_0.74.pth', map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError


    return model, optimizer

def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device=torch.device("cpu")
    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.to(device)
            output = model(img)
            output = output.squeeze()
            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            mask = np.zeros_like(output[0])
            for c in range(14):
                if c == 0:
                    continue
                mask[np.where(output[c] == 1)] = c
            mask = mask.astype('uint8')
            save_name=os.path.join(args.pred_mask_path,os.path.basename(id[0]))
            nrrd.write(save_name, mask)

def read_img(path):
    img=sitk.ReadImage(path)
    data=sitk.GetArrayFromImage(img)
    return data

def normalization_imgs(imgs):
    ''' centering and reducing data structures '''
    imgs = imgs.astype(np.float32, copy=False)
    mean = np.mean(imgs) # mean for data centering
    std = np.std(imgs) # std for data normalization
    if np.int32(std) != 0:
        imgs -= mean
        imgs /= std
    return imgs

def read_nrrd(path):
    img=nrrd.read(path)
    data=img[0]
    return data


def save(path, name, save_path):
    data = read_img(os.path.join(path, name))
    data = normalization_imgs(data)
    D, W, H = data.shape
    name1 = name.split('.')[0]
    if name1.endswith('0000'):
        name1 = name1[:-4]
    for i in range(D):
        img_gray = data[i]
        name2 = name1 + '_' + str(i + 1).zfill(4) + '.nrrd'
        save_name = os.path.join(save_path, name2)
        nrrd.write(save_name, img_gray)


def save_nii(path, ids, save_path):
    mask_list = []
    for i in ids:
        img = os.path.join(path, i)
        mask = read_nrrd(img)
        mask_list.append(mask)

    fin = np.dstack(mask_list)
    fin = fin.transpose(2, 0, 1)
    s = ids[0].split('_')
    save_name = os.path.join(save_path, s[0] + '_' + s[1] + '.nii.gz')
    save_img = sitk.GetImageFromArray(fin)
    sitk.WriteImage(save_img, save_name)

def prepare():
    img_path = "./inputs"
    save_path = "./dataset/val"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    names = os.listdir(img_path)
    for n in names:
        save(img_path, n, save_path)
    unlabeled_images = os.listdir(save_path)
    with open("dataset/splits/pred.txt", 'w') as f:
        for i in range(len(unlabeled_images)):
            f.write(unlabeled_images[i] + '\n')
            i += 1
def end():
    labels_path = "./pred_mask"
    labels = os.listdir(labels_path)
    old = int(labels[0].split('_')[1])
    tmp = []
    label_list = []
    for i in labels:
        if int(i.split('_')[1]) > old:
            label_list.append(tmp)
            old += 1
            tmp = []
        tmp.append(i)
        if i == labels[-1]:
            label_list.append(tmp)
    save_path = "./outputs"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for labels in label_list:
        save_nii(labels_path, labels, save_path)

def main(args):
    if not os.path.exists(args.pred_mask_path):
        os.makedirs(args.pred_mask_path)
    # thread1=threading.Thread(target=get_used_cpu)
    # thread2=threading.Thread(target=get_free_gpu_number)
    # thread1.start()
    # thread2.start()
    prepare()
    cur_unlabeled_id_path= 'dataset/splits/pred.txt'

    dataset = SemiDataset(args.data_root, 'label', args.num_class, None, cur_unlabeled_id_path,
                          transforms=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    model, optimizer = init_basic_elems(args)

    label(model, dataloader, args)

    end()
if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    main(args)
    end_time=time.time()
    print(end_time-start_time)