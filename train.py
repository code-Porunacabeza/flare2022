from dataset.dataset_falre import SemiDataset
from utils import count_params, iou_score,iou_dice
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations import RandomRotate90,Resize
from losses import BCEDiceLoss
from model import archs

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import nrrd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='FLARE2022')

    # basic settings
    parser.add_argument('--data-root', type=str, default='/root/autodl-tmp/FLARE2022/train')
    parser.add_argument('--dataset', type=str, default='flare')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--num-class', type=int, default=14)



    parser.add_argument('--model', type=str,
                        default='Unext')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, default='./dataset/splits/labeled.txt')
    parser.add_argument('--unlabeled-id-path', type=str, default='./dataset/splits/unlabeled.txt')
    parser.add_argument('--pseudo-mask-path', type=str,default='./pseudo_mask')

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

def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    criterion = BCEDiceLoss().cuda()
    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),


    ])


    valset = SemiDataset(args.data_root, 'val',args.num_class, args.labeled_id_path)
    valloader = DataLoader(valset, batch_size=16,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    global MODE
    MODE = 'train'


    trainset = SemiDataset(args.data_root, MODE, args.num_class, args.labeled_id_path,transforms=train_transform)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))

    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)


    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')
    #选取可靠图像
    dataset = SemiDataset(args.data_root, 'label',args.num_class, None, args.unlabeled_id_path,transforms=train_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')
    #给可靠图像打伪标签
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.data_root, 'label',args.num_class, None, cur_unlabeled_id_path,transforms=train_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')
    #第一次在伪标签上训练
    MODE = 'semi_train'

    trainset = SemiDataset(args.data_root, MODE, args.num_class,
                           args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path,transforms=train_transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)

    best_model = train(model, trainloader, valloader, criterion, optimizer, args)

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')
    #给不可靠图像打伪标签
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    dataset = SemiDataset(args.data_root, 'label',args.num_class, None, cur_unlabeled_id_path,transforms=train_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')
    # 第二次在伪标签上训练
    trainset = SemiDataset(args.data_root, MODE, args.num_class,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path,transforms=train_transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, criterion, optimizer, args)


def init_basic_elems(args):
    # create model
    config = vars(parse_args2())
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels']
                                           )

    model = model.cuda()

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


def train(model, trainloader, valloader, criterion, optimizer, args):
    config = vars(parse_args2())
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    global MODE

    if MODE == 'train':
        checkpoints = []

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            #lr = args.lr * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        # metric = meanIOU(num_classes=14)
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        model.eval()
        tbar = tqdm(valloader)
        total_dice=[]
        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                mask=mask.cuda()
                pred = model(img)

                mIOU,dice = iou_dice(pred,mask)
                total_dice.append(dice)
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                tbar.set_description('dice: %.2f' % (dice))

        mIOU *= 100.0

        mean_dice=sum(total_dice)/len(total_dice)
        if mean_dice > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (args.model, previous_best)))
            previous_best = mean_dice
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, '%s_%.2f.pth' % (args.model, mean_dice)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
        # if MODE == 'train' :
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model

# def change_mask(mask):
#     preds=[]
#     for m in mask:
#         f_mask = np.zeros_like(m[0])
#         for c in range(14):
#             if c == 0:
#                 continue
#             f_mask[np.where(m[c] == 1)] = c
#         f_mask=f_mask[np.newaxis,:,:]
#         preds.append(f_mask)
#         arr = np.dstack(preds)
#         # fu_mask = f_mask.astype('uint8')
#     return arr.transpose(3, 0, 1, 2)

def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(model(img).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                mIOU.append(iou_score(preds[i], preds[-1]))

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)


    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
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
            save_name=os.path.join(args.pseudo_mask_path,os.path.basename(id[0]))
            nrrd.write(save_name, mask)


            # pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))
            #
            # tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()
    print(args)

    main(args)
