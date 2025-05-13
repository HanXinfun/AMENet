import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os

# Parameters
## SwinFusion
WINDOW_SIZE = (256, 256) # Patch size
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)


FOLDER = "/public/data/hxx/dataset/" # /data/code/dataset/
BATCH_SIZE = 10 # Number of samples in a mini-batch ########## hjx  ori: 10

CHOOSE_DATA = 0     ########## hjx 更换数据集 ==>  0:Vaihingen，1:Potsdam，2:CBMD，3:whu-opt-sar
MODE = "train"      ########## hjx 训练/测试 train/test
EPOCHS = 50      ########## hjx 
VSS_ON = True      ########## hjx 是否使用 VSSBlock True/False
EXPERTS_ON = False      ########## hjx 是否使用 多专家 True/False

"""
对比模型 hjx 
AMENet/FTransUNet/TransUNet/MambaUNet/VMUNet/ 
ABCNet/CMGFNet/ESANet/FuseNet/MAResUNet/SAGATE/UNetFormer/MANet/DCSwin/PPMamba
"""
COMPARE_MODEL = "AMENet"      


############    Vaihingen 数据集    ############

if CHOOSE_DATA == 0:
    LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names ##
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    CACHE = True # Store the dataset in-memory
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30']
    DATASET = 'Vaihingen'
    Stride_Size = 32 # Stride for testing
    MAIN_FOLDER = FOLDER + 'Vaihingen/semantic/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

############    Vaihingen 数据集    ############




############    Potsdam 数据集    ############

if CHOOSE_DATA == 1:
    LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    CACHE = True # Store the dataset in-memory
    train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
                '4_12', '6_8', '6_12', '6_7', '4_11']
    test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
    DATASET = 'Potsdam'
    Stride_Size = 128 # 128 for quickly # Stride for testing
    MAIN_FOLDER = FOLDER + 'Potsdam/semantic/'
    DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
    DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation_rename/dsm_potsdam_{}_normalized_lastools.jpg'      #### hjx , 原文件夹 1_DSM_normalisation 是 01-09，用 rename.py
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'

############    Potsdam 数据集    ############





############    CBMD 数据集    ############

if CHOOSE_DATA == 2:
    LABELS = ["pyroclastic deposits", "mudflow deposits", "empty deposits", "basaltic lava platform", "trachyte", "tongue"] # Label names
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    CACHE = True # Store the dataset in-memory
    train_ids = ['4_3_10', '5_2_5', '4_2_1', '6_4_7', '2_3_0', '1_3_1', '2_1_8', '1_6_8', '4_2_9', '4_2_0', '4_4_10', '4_1_3', '6_5_0', '3_3_1', '1_5_4', '3_4_8', '2_4_5', '2_6_3', '5_3_8', '1_5_5', '1_5_6', '6_4_10', '2_6_2', '1_4_10', '2_4_8', '5_2_2', '5_1_0', '1_4_3', '1_6_7', '1_2_7', '1_6_4', '3_6_4', '3_5_7', '1_1_6', '6_5_4', '4_1_0', '3_1_10', '2_2_2', '2_2_10', '1_4_4', '2_4_0', '1_6_6', '3_1_8', '5_4_6', '1_1_2', '4_1_2', '1_4_7', '1_4_2', '1_2_8', '2_5_1', '3_6_6', '4_3_0', '2_1_6', '4_2_6', '1_3_8', '3_3_5', '1_1_1', '3_2_10', '1_3_3', '5_1_7', '6_5_2', '3_6_0', '1_3_7', '6_3_4', '6_1_5', '1_3_6', '5_3_7', '6_2_7', '6_2_6', '2_6_7', '2_6_0', '5_1_8', '2_6_6', '5_4_9', '4_4_1', '2_4_6', '3_6_8', '2_3_9', '6_3_1', '6_1_8', '5_3_5', '1_5_8', '3_2_9', '5_4_1', '1_1_5', '3_4_1', '1_1_10', '3_3_7', '1_6_10', '2_6_9', '3_2_4', '3_1_5', '6_3_7', '3_2_1', '4_1_4', '4_2_10', '3_5_10', '6_5_10', '5_1_9', '6_2_10', '2_2_7', '4_4_3', '3_1_7', '5_2_6', '4_2_7', '3_1_4', '2_5_8', '5_1_3', '1_2_0', '5_2_7', '3_5_2', '6_3_6', '6_2_0', '3_6_5', '2_1_9', '6_3_8', '1_2_1', '2_5_3', '3_4_2', '6_5_8', '3_1_2', '6_2_5', '3_2_0', '4_4_0', '2_5_2', '2_4_4', '6_4_8', '6_4_0', '5_4_4', '5_1_1', '1_5_9', '2_4_10', '3_2_5', '2_6_1', '3_4_4', '4_4_9', '2_2_1', '2_5_5', '6_4_9', '1_1_9', '1_4_1', '5_1_6', '3_5_4', '6_3_10', '5_4_0', '5_2_4', '4_3_3', '4_3_4', '6_5_6', '1_4_8', '1_4_5', '3_5_0', '6_4_6', '1_4_0', '2_3_7', '5_4_8', '6_4_3', '2_4_2', '2_3_8', '6_3_2', '1_5_0', '2_1_2', '3_6_9', '3_3_10', '4_3_7', '3_1_1', '2_4_1', '3_3_2', '6_5_5', '6_2_4', '3_1_3', '2_1_4', '5_1_10', '2_2_4', '5_2_10', '5_1_4', '2_2_8', '2_4_7', '2_3_2', '6_1_7', '4_3_1', '5_2_0', '4_3_8', '6_2_8', '2_1_10', '2_2_9', '2_1_0', '1_3_4', '3_5_6', '5_2_8', '4_4_6', '2_2_5', '3_2_2', '2_4_3', '2_6_5', '1_2_2', '2_5_0', '5_3_9', '4_3_5', '6_4_1', '6_1_1', '6_4_2', '2_3_3', '3_1_6', '6_5_9', '2_3_6', '1_5_2', '4_1_5', '6_3_3', '5_3_10', '6_1_9', '2_2_3', '1_6_3', '4_4_8', '3_4_5', '1_2_5', '1_1_7', '1_6_1', '1_5_3', '1_2_10', '2_5_6', '5_4_5', '3_6_2', '3_4_7', '2_1_5', '5_3_1', '6_1_3', '1_5_7', '3_3_0', '2_3_4', '3_5_3', '1_5_1', '6_3_5', '2_1_3', '1_6_0', '3_2_3', '3_6_3', '2_5_10']
    test_ids = ['6_1_10', '5_3_4', '4_1_8', '3_3_4', '1_3_2', '2_1_7', '5_2_9', '2_2_6', '4_3_9', '2_3_1', '1_6_2', '1_6_9', '2_3_10', '4_1_6', '4_2_5', '3_4_6', '3_6_1', '4_2_8', '2_4_9', '3_3_3', '2_6_10', '1_1_3', '1_3_5', '3_5_5', '2_5_4', '1_2_6', '6_2_9', '3_2_6', '3_5_9', '3_2_7', '6_1_0', '1_6_5', '3_3_8', '1_5_10', '3_2_8', '6_5_7', '6_1_4', '4_1_7', '1_3_10', '6_3_0', '1_1_8', '4_1_10', '1_1_0', '6_5_1', '1_2_3', '4_3_2', '5_3_0', '1_4_9', '4_2_3', '5_2_3', '5_2_1', '3_4_10', '6_2_3', '2_2_0', '5_3_3', '2_3_5', '2_5_7', '3_4_0', '5_4_3', '4_4_4', '3_1_0', '3_3_6', '3_5_1', '5_4_2', '5_4_10', '3_4_9', '4_4_2', '4_4_7', '6_4_5', '5_3_2', '3_3_9', '6_1_2', '1_3_9', '1_2_4', '6_4_4', '5_4_7', '3_4_3', '6_1_6', '2_1_1', '4_1_9', '2_6_4', '6_3_9', '6_5_3', '4_1_1', '1_4_6', '2_6_8', '4_2_4', '6_2_1', '1_2_9', '2_5_9', '3_6_7', '5_3_6', '5_1_5', '3_5_8', '4_2_2', '5_1_2', '1_3_0', '4_4_5', '4_3_6', '1_1_4', '3_1_9', '3_6_10', '6_2_2']
    DATASET = 'Vaihingen'
    Stride_Size = 32 # Stride for testing
    MAIN_FOLDER = FOLDER + 'CBMD_aug/'
    DATA_FOLDER = MAIN_FOLDER + 'Image/output_{}.tif'
    DSM_FOLDER = MAIN_FOLDER + 'Dem/output_{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'label/output_{}.png'
    ERODED_FOLDER = MAIN_FOLDER + 'label_with_borders/output_{}.png'

############    CMLD 数据集    ############





############    whu-opt-sar 数据集    ############

if CHOOSE_DATA == 3:
    LABELS = ["farmland", "city", "village", "water", "forest", "road", "others"] # Label names
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    CACHE = True # Store the dataset in-memory
    train_ids = ['13007', '14021', '12020', '07013', '10016', '12018', '03024', '05011', '05024', '05023', '09017', '06024', '13010', '06022', '01014', '10022', '13021', '06017'] # , '10020', '08024', '10014', '14024', '07014', '07023', '12009', '08013', '09012', '11017', '09024', '04011', '14022', '06011', '11024', '14017', '02016', '08014', '13018', '14007', '03014', '06019', '05014']
    test_ids = ['04021', '08017', '02014', '06014', '02012', '01013'] # '13023', '07024', '01017', '05019', '10015', '12022', '09014', '07017', '04013', '11011', '12017', '09023']
    DATASET = 'Potsdam'     ###  whu-opt 可见光四通道数据（B,G,R,NIR）,使用 Potsdam 的数据加载方法，取 RGB 的三通道
    Stride_Size = 128 # 128 for quickly # Stride for testing
    MAIN_FOLDER = FOLDER + 'whu-opt-sar/'
    DATA_FOLDER = MAIN_FOLDER + 'optical/NH49E0{}.tif'
    DSM_FOLDER = MAIN_FOLDER + 'sar/NH49E0{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'predict/NH49E0{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'predict/NH49E0{}.tif'

############    whu-opt-sar 数据集    ############




############    ISPRS color palette      ############
if CHOOSE_DATA == 0:
    palette = {0 : (255, 255, 255), # Impervious surfaces (white)
            1 : (0, 0, 255),     # Buildings (blue)
            2 : (0, 255, 255),   # Low vegetation (cyan)
            3 : (0, 255, 0),     # Trees (green)
            4 : (255, 255, 0),   # Cars (yellow)
            5 : (255, 0, 0),     # Clutter (red)  # 背景类
            6 : (0, 0, 0)}       # Undefined (black)

if CHOOSE_DATA == 1:
    palette = {0 : (255, 255, 255), # Impervious surfaces (white)
            1 : (0, 0, 255),     # Buildings (blue)
            2 : (0, 255, 255),   # Low vegetation (cyan)
            3 : (0, 255, 0),     # Trees (green)
            4 : (255, 255, 0),   # Cars (yellow)
            5 : (255, 0, 0),     # Clutter (red)
            6 : (0, 0, 0)}       # Undefined (black)
############    ISPRS color palette      ############



############    CMLD color palette      ############
if CHOOSE_DATA == 2:
    palette = {0 : (255, 190, 0),   # Pyroclastic accumulation （碎屑）
            1 : (255, 225, 136), # Lahar accumulation (泥流)
            2 : (255, 206, 163), # Ashfall accumulation （空落）
            3 : (1, 176, 0),     # basalt（玄武岩）
            4 : (255, 135, 0),   # trachyte（粗面岩）
            5 : (191, 99, 0)}    # Pantellerite（碱流岩）
            # 6 : (0, 0, 0)}       # Undefined (black)
############    CMLD color palette      ############



############    whu-opt-sar color palette      ############
if CHOOSE_DATA == 3:
    palette = {0 : (204,102,0),   # farmland
            1 : (255,0,0), # city
            2 : (255,255,0), # village
            3 : (0,0,255),     # water
            4 : (85,167,0),   # forest 166->167
            5 : (0,255,255),   # road 93->0
            6 : (153,102,153)}    # others 152(1)->153
            # 7 : (0, 0, 0)}       # Undefined (black)
############    whu-opt-sar color palette      ############



invert_palette = {v: k for k, v in palette.items()}






def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """

    ########### hjx ###########
    # 改：将 png 图像转换为 3 通道，去除 alpha 透明度通道
    if CHOOSE_DATA == 2:
        arr_3d = arr_3d[:, :, :3]  
    ########### hjx ###########

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')



class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        # self.boundary_files = [BOUNDARY_FOLDER.format(id) for id in ids]
        self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        # self.boundary_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        if DATASET == 'Potsdam':
            return BATCH_SIZE * 1000
        elif DATASET == 'Vaihingen':
            return BATCH_SIZE * 1000
        else:
            return None

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            ## Potsdam IRRG
            if DATASET == 'Potsdam':
                ## RGB
                data = io.imread(self.data_files[random_idx])[:, :, :3].transpose((2, 0, 1))
                ## IRRG
                # data = io.imread(self.data_files[random_idx])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                data = 1 / 255 * np.asarray(data, dtype='float32')
            else:
            ## Vaihingen IRRG
                data = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
        
        # if random_idx in self.boundary_cache_.keys():
        #     boundary = self.boundary_cache_[random_idx]
        # else:
        #     boundary = np.asarray(io.imread(self.boundary_files[random_idx])) / 255
        #     boundary = boundary.astype(np.int64)
        #     if self.cache:
        #         self.boundary_cache_[random_idx] = boundary

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')
            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        # boundary_p = boundary[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        # data_p, boundary_p, label_p = self.data_augmentation(data_p, boundary_p, label_p)
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))
        


## We load one tile from the dataset and we display it
# img = io.imread('./ISPRS_dataset/Vaihingen/top/top_mosaic_09cm_area11.tif')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)
#
# # We load the ground truth
# gt = io.imread('./ISPRS_dataset/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
# fig.add_subplot(122)
# plt.imshow(gt)
# plt.show()
#
# # We also check that we can convert the ground truth into an array format
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)



# Utils

# 根据 window_shape 随机获取位置
def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]

    ### hjx ###
    # 如果图像大小和窗口大小相同，直接返回整个图像的坐标
    if W == w and H == h:
        return 0, W, 0, H   # CHOOSE_DATA == 2 (256,256)
    ### hjx ###
    
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return accuracy
