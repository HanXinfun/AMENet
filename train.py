import numpy as np
from glob import glob
# from tqdm import tqdm_notebook as tqdm
# from tqdm.notebook import tqdm
from tqdm.auto import tqdm

from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output

from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg


#### hjx ####
num_workers = 2 # ori:4
torch.set_num_threads(num_workers)

if COMPARE_MODEL == "AMENet" and VSS_ON == True and EXPERTS_ON == True:
    from model.AMENet import AMENet
if COMPARE_MODEL == "AMENet" and VSS_ON == True and EXPERTS_ON == False:
    from model.AMENet_vss import AMENet
if COMPARE_MODEL == "AMENet" and VSS_ON == False and EXPERTS_ON == True:
    from model.AMENet_me import AMENet
if COMPARE_MODEL == "AMENet" and VSS_ON == False and EXPERTS_ON == False:
    from model.AMENet_no import AMENet


if COMPARE_MODEL == "FTransUNet":
    from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
if COMPARE_MODEL == "TransUNet":
    from compare_model.TransUNet.TransUNet import VisionTransformer
if COMPARE_MODEL == "MambaUNet":
    from compare_model.MambaUNet.MambaUNet import MambaUNet
if COMPARE_MODEL == "VMUNet":
    from compare_model.VMUNet.VMUNet import VMUNet

if COMPARE_MODEL == "ABCNet":
    from compare_model.ABCNet.ABCNet import ABCNet
if COMPARE_MODEL == "CMGFNet":
    from compare_model.CMGFNet.CMGFNet import CMGFNet
if COMPARE_MODEL == "ESANet":
    from compare_model.ESANet.ESANet import ESANet
if COMPARE_MODEL == "FuseNet":
    from compare_model.FuseNet.FuseNet import FuseNet
if COMPARE_MODEL == "MAResUNet":
    from compare_model.MAResUNet.MAResUNet import MAResUNet
if COMPARE_MODEL == "SAGATE":
    from compare_model.SAGATE.SAGATE import SAGATE
if COMPARE_MODEL == "UNetFormer":
    from compare_model.UNetFormer.UNetFormer import UNetFormer
if COMPARE_MODEL == "MANet":
    from compare_model.UNetFormer.othermodels.MANet import MANet
if COMPARE_MODEL == "DCSwin":
    from compare_model.UNetFormer.othermodels.DCSwin import *
if COMPARE_MODEL == "PPMamba":
    from compare_model.UNetFormer.othermodels.PPMamba import PPMamba
#### hjx ####



try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

## model/vitcross_seg_modeling.py 中映射 
## R50-ViT-B_16 -> get_r50_b16_config
config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']    
config_vit.n_classes = N_CLASSES
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16)) 

## 模型
if COMPARE_MODEL == "AMENet":
    net = AMENet(num_classes=N_CLASSES).cuda()
if COMPARE_MODEL == "FTransUNet":
    net = ViT_seg(config_vit, img_size=256, num_classes=N_CLASSES).cuda() 
if COMPARE_MODEL == "TransUNet":
    net = VisionTransformer(config_vit, img_size=256, num_classes=N_CLASSES).cuda() 
if COMPARE_MODEL == "MambaUNet":
    import argparse
    from compare_model.MambaUNet.MambaUNet_config import get_config
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MambaUnet for Image Segmentation")
    parser.add_argument('--cfg', type=str, default="/public/data/hxx/SSRS/FTransUNet/compare_model/MambaUNet/vmamba_tiny.yaml", help='path to config file', )
    args = parser.parse_args()
    # 加载配置文件
    config = get_config(args)
    # 初始化模型
    net = MambaUNet(config, img_size=256, num_classes=N_CLASSES).cuda()
    net.load_from(config=config)
if COMPARE_MODEL == "VMUNet":
    net = VMUNet(input_channels=3,num_classes=N_CLASSES,load_ckpt_path="/public/data/hxx/SSRS/FTransUNet/compare_model/VMUNet/vmamba_base_224.pth").cuda() # vmamba_tiny_e292
    net.load_from()
if COMPARE_MODEL == "ABCNet":
    net = ABCNet(band=3, n_classes=N_CLASSES).cuda()
if COMPARE_MODEL == "CMGFNet":
    net = CMGFNet(num_classes=N_CLASSES).cuda()
if COMPARE_MODEL == "ESANet":
    net = ESANet(height=256,width=256,num_classes=N_CLASSES).cuda()
if COMPARE_MODEL == "FuseNet":
    net = FuseNet(num_labels=N_CLASSES,use_class=False).cuda()
if COMPARE_MODEL == "MAResUNet":
    net = MAResUNet(num_channels=3, num_classes=N_CLASSES).cuda()
if COMPARE_MODEL == "SAGATE":
    net = SAGATE(out_planes=N_CLASSES, criterion=nn.CrossEntropyLoss(),pretrained_model=None,norm_layer=nn.BatchNorm2d).cuda()
if COMPARE_MODEL == "UNetFormer":
    net = UNetFormer(num_classes=N_CLASSES).cuda()
if COMPARE_MODEL == "MANet":
    net = MANet(num_classes=N_CLASSES,num_channels=3,backbone_name='resnet50').cuda()
if COMPARE_MODEL == "DCSwin":
    net = dcswin_tiny(num_classes=N_CLASSES,pretrained=False).cuda()
if COMPARE_MODEL == "PPMamba":
    net = PPMamba(num_classes=N_CLASSES).cuda()

# net.load_from(weights=np.load(config_vit.pretrained_path))


#### hjx ####
# CMBD 不加载预训练权重
# if CHOOSE_DATA != 2:
#     net.load_from(weights=np.load(config_vit.pretrained_path))
#### hjx ####


params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)
# Load the datasets

print("training : ", train_ids)
print("testing : ", test_ids)
print("BATCH_SIZE: ", BATCH_SIZE)
print("Stride Size: ", Stride_Size)
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE, num_workers=num_workers) ##### hjx：num_workers

base_lr = 0.01 ##### hjx：0.01

params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)  # [25, 35, 45]


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    ## Potsdam
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)

                # Do the inference
                # outs = net(image_patches, dsm_patches)

                ### hjx ###
                if COMPARE_MODEL == "AMENet" or COMPARE_MODEL == "FTransUNet" \
                or COMPARE_MODEL == "CMGFNet" or COMPARE_MODEL == "ESANet" or COMPARE_MODEL == "FuseNet" \
                or COMPARE_MODEL == "SAGATE":
                    outs = net(image_patches, dsm_patches)
                if COMPARE_MODEL == "TransUNet" or COMPARE_MODEL == "MambaUNet" or COMPARE_MODEL == "VMUNet" \
                or COMPARE_MODEL == "ABCNet" or COMPARE_MODEL == "MAResUNet" or COMPARE_MODEL == "UNetFormer" \
                or COMPARE_MODEL == "MANet" or COMPARE_MODEL == "DCSwin" or COMPARE_MODEL == "PPMamba":
                    outs = net(image_patches)
                ### hjx ###
                ### hjx ###
                if COMPARE_MODEL == "AMENet" or COMPARE_MODEL == "CMGFNet":
                    outs = outs[0]
                ### hjx ###
                
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
            
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):   # WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    criterion = nn.NLLLoss2d(weight=weights)
    iter_ = 0

    # acc_best = 90.0     
    ### hjx ###
    if COMPARE_MODEL == "MambaUNet" or COMPARE_MODEL == "VMUNet":
        acc_best = 84.0   
    else:
        acc_best = 86.0     

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())

            # ### hjx ###
            # # 检查 target 的范围
            # print(f"data.shape:",data.shape)
            # print(f"dsm.shape:",dsm.shape)
            # print(f"target.shape:",target.shape)
            # print(f"Batch {batch_idx}: target min={target.min().item()}, max={target.max().item()}")
            # ### hjx ###
          
            optimizer.zero_grad()

            # output = net(data, dsm)

            ### hjx ###
            if COMPARE_MODEL == "AMENet" or COMPARE_MODEL == "FTransUNet" \
            or COMPARE_MODEL == "CMGFNet" or COMPARE_MODEL == "ESANet" or COMPARE_MODEL == "FuseNet" \
            or COMPARE_MODEL == "SAGATE":
                output = net(data, dsm)
            if COMPARE_MODEL == "TransUNet" or COMPARE_MODEL == "MambaUNet" or COMPARE_MODEL == "VMUNet" \
            or COMPARE_MODEL == "ABCNet" or COMPARE_MODEL == "MAResUNet" or COMPARE_MODEL == "UNetFormer" \
            or COMPARE_MODEL == "MANet" or COMPARE_MODEL == "DCSwin" or COMPARE_MODEL == "PPMamba":
                output = net(data)
            ### hjx ###

            if COMPARE_MODEL == "ABCNet":
                output = output[0]+output[1]+output[2]
                loss = F.cross_entropy(output, target, weight=weights)
            if COMPARE_MODEL == "CMGFNet" or COMPARE_MODEL == "ESANet" or COMPARE_MODEL == "UNetFormer":
                output = output[0]
                loss = F.cross_entropy(output, target, weight=weights)
            ### hjx ###
            if COMPARE_MODEL == "AMENet":
                output_aux = output[1]
                output = output[0]
                ce_loss = CrossEntropy2d(output, target, weight=weights)
                di_loss = dice_loss(output, target)
                aux_ce_loss = CrossEntropy2d(output_aux, target, weight=weights)
                loss = ce_loss + di_loss + 0.4 * aux_ce_loss
            else:
                loss = CrossEntropy2d(output, target, weight=weights)
            ### hjx ###

            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

            # if e % save_epoch == 0:
            if iter_ % 500 == 0:
                net.eval()
                acc = test(net, test_ids, all=False, stride=Stride_Size)
                net.train()
                if acc > acc_best:
                    # torch.save(net.state_dict(), './resultsv_se_ablation/epoch{}_{}'.format(e, acc))

                    ######### hjx #########

                    if COMPARE_MODEL == "AMENet":
                        if CHOOSE_DATA == 0:
                            if VSS_ON == True and EXPERTS_ON == False:
                                torch.save(net.state_dict(), 'result_AMENet_vss/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                            if VSS_ON == False and EXPERTS_ON == True:
                                torch.save(net.state_dict(), 'result_AMENet_me/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                            if VSS_ON == False and EXPERTS_ON == False:
                                torch.save(net.state_dict(), 'result_AMENet_no/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                            if VSS_ON == True and EXPERTS_ON == True:
                                torch.save(net.state_dict(), 'result_AMENet_vss_me/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            if VSS_ON == True and EXPERTS_ON == False:
                                torch.save(net.state_dict(), 'result_AMENet_vss/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                            if VSS_ON == False and EXPERTS_ON == True:
                                torch.save(net.state_dict(), 'result_AMENet_me/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                            if VSS_ON == False and EXPERTS_ON == False:
                                torch.save(net.state_dict(), 'result_AMENet_no/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                            if VSS_ON == True and EXPERTS_ON == True:
                                torch.save(net.state_dict(), 'result_AMENet_vss_me/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            if VSS_ON == True and EXPERTS_ON == False:
                                torch.save(net.state_dict(), 'result_AMENet_vss/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                            if VSS_ON == False and EXPERTS_ON == True:
                                torch.save(net.state_dict(), 'result_AMENet_me/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                            if VSS_ON == False and EXPERTS_ON == False:
                                torch.save(net.state_dict(), 'result_AMENet_no/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                            if VSS_ON == True and EXPERTS_ON == True:
                                torch.save(net.state_dict(), 'result_AMENet_vss_me/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            if VSS_ON == True and EXPERTS_ON == False:
                                torch.save(net.state_dict(), 'result_AMENet_vss/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))
                            if VSS_ON == False and EXPERTS_ON == True:
                                torch.save(net.state_dict(), 'result_AMENet_me/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))
                            if VSS_ON == False and EXPERTS_ON == False:
                                torch.save(net.state_dict(), 'result_AMENet_no/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))
                            if VSS_ON == True and EXPERTS_ON == True:
                                torch.save(net.state_dict(), 'result_AMENet_vss_me/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "FTransUNet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_FTransUnet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_FTransUnet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_FTransUnet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_FTransUnet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "TransUNet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_TransUNet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_TransUNet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_TransUNet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_TransUNet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "MambaUNet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_MambaUNet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_MambaUNet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_MambaUNet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_MambaUNet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "VMUNet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_VMUNet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_VMUNet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_VMUNet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_VMUNet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "ABCNet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_ABCNet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_ABCNet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_ABCNet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_ABCNet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "CMGFNet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_CMGFNet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_CMGFNet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_CMGFNet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_CMGFNet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "ESANet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_ESANet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_ESANet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_ESANet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_ESANet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "FuseNet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_FuseNet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_FuseNet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_FuseNet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_FuseNet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "MAResUNet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_MAResUNet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_MAResUNet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_MAResUNet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_MAResUNet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "SAGATE":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_SAGATE/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_SAGATE/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_SAGATE/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_SAGATE/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "UNetFormer":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_UNetFormer/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_UNetFormer/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_UNetFormer/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_UNetFormer/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "MANet":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_MANet/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_MANet/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_MANet/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_MANet/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))
                            
                    if COMPARE_MODEL == "DCSwin":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_DCSwin/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_DCSwin/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_DCSwin/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_DCSwin/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    if COMPARE_MODEL == "PPMamba":
                        if CHOOSE_DATA == 0:
                            torch.save(net.state_dict(), 'result_PPMamba/resultsv_Vaihingen/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 1:
                            torch.save(net.state_dict(), 'result_PPMamba/resultsv_Potsdam/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 2:
                            torch.save(net.state_dict(), 'result_PPMamba/resultsv_CBMD/epoch{}_{}'.format(e, acc))
                        if CHOOSE_DATA == 3:
                            torch.save(net.state_dict(), 'result_PPMamba/resultsv_whu-opt-sar/epoch{}_{}'.format(e, acc))

                    ######### hjx #########
                    
                    acc_best = acc

    print('acc_best: ', acc_best)



######### hjx #########
def dice_loss(pred, target, epsilon=1e-6):
    """
    pred: [B, C, H, W] (logits)，需要先做 softmax
    target: [B, H, W] (整型标签)
    """
    pred_soft = F.softmax(pred, dim=1)  # [B, C, H, W]
    # 转成one-hot: [B, C, H, W]
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()
    
    dims = (0, 2, 3)
    intersection = torch.sum(pred_soft * target_onehot, dims)
    cardinality = torch.sum(pred_soft**2 + target_onehot**2, dims)
    dice_score = (2. * intersection + epsilon) / (cardinality + epsilon)
    return 1. - dice_score.mean()
######### hjx #########



#####   train   ####
if MODE == "train":
    time_start=time.time()
    train(net, optimizer, EPOCHS, scheduler)
    time_end=time.time()
    print('Total Time Cost: ',time_end-time_start)



