import os

#
# path and dataset parameter
#
# 第一部分是使用到的数据相关参数，包括数据路径预训练权重等相关内容。
DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')

WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

# 是否对样本图像进行flip（水平镜像）操作
FLIPPED = True


#
# model parameter
#
# 这部分主要是模型参数
# 图像size
IMAGE_SIZE = 448

# 网格 size
CELL_SIZE = 7

# 每个 cell 中 bounding box 数量
BOXES_PER_CELL = 2

# 权重衰减相关参数
ALPHA = 0.1

DISP_CONSOLE = False

# 权重衰减的相关参数
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#
# 训练过程中的相关参数
GPU = ''

# 学习速率
LEARNING_RATE = 0.0001

# 衰减步数
DECAY_STEPS = 30000

# 衰减率
DECAY_RATE = 0.1

STAIRCASE = True

# batch_size初始值为45
BATCH_SIZE = 45

# 最大迭代次数
MAX_ITER = 15000

# 日志记录迭代步数
SUMMARY_ITER = 10

SAVE_ITER = 1000


#
# test parameter
#
# 测试时的相关参数

# 阈值参数
THRESHOLD = 0.2

# IoU 参数
IOU_THRESHOLD = 0.5
