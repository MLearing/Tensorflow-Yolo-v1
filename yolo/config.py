import os

#
# path and dataset parameter
#
# ��һ������ʹ�õ���������ز�������������·��Ԥѵ��Ȩ�ص�������ݡ�
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

# �Ƿ������ͼ�����flip��ˮƽ���񣩲���
FLIPPED = True


#
# model parameter
#
# �ⲿ����Ҫ��ģ�Ͳ���
# ͼ��size
IMAGE_SIZE = 448

# ���� size
CELL_SIZE = 7

# ÿ�� cell �� bounding box ����
BOXES_PER_CELL = 2

# Ȩ��˥����ز���
ALPHA = 0.1

DISP_CONSOLE = False

# Ȩ��˥������ز���
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#
# ѵ�������е���ز���
GPU = ''

# ѧϰ����
LEARNING_RATE = 0.0001

# ˥������
DECAY_STEPS = 30000

# ˥����
DECAY_RATE = 0.1

STAIRCASE = True

# batch_size��ʼֵΪ45
BATCH_SIZE = 45

# ����������
MAX_ITER = 15000

# ��־��¼��������
SUMMARY_ITER = 10

SAVE_ITER = 1000


#
# test parameter
#
# ����ʱ����ز���

# ��ֵ����
THRESHOLD = 0.2

# IoU ����
IOU_THRESHOLD = 0.5
