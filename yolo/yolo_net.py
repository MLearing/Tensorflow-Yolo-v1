import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim

"""
参考链接：
1.https://blog.csdn.net/qq_34784753/article/details/78803423
2.https://blog.csdn.net/qq1483661204/article/details/79681926
3.https://blog.csdn.net/c20081052/article/details/80260726
"""

class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES         #类别
        self.num_class = len(self.classes) #类别数 20
        self.image_size = cfg.IMAGE_SIZE   #图片尺寸 448
        self.cell_size = cfg.CELL_SIZE     #cell大小 7*7
        self.boxes_per_cell = cfg.BOXES_PER_CELL #每个 cell 中 bounding box 数量 2
        self.output_size = (self.cell_size * self.cell_size) *\   #输出tensor大小[7*7*(20+2*5)]
            (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size  # 单个cell尺度
        self.boundary1 = self.cell_size * self.cell_size * self.num_class #所有的 cell 的类别的预测的张量维度[7*7*20]
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell  #在类别之后每个cell 所对应的 bounding boxes 的数量的总和[7*7*(20+2)]

        # 损失函数中的相关lamda参数
        self.object_scale = cfg.OBJECT_SCALE     # 值为1，有目标存在的系数   
        self.noobject_scale = cfg.NOOBJECT_SCALE # λnoobj
        self.class_scale = cfg.CLASS_SCALE       # 值为2.0， 类别损失函数的系数
        self.coord_scale = cfg.COORD_SCALE       # λcoord

        self.learning_rate = cfg.LEARNING_RATE  #学习率
        self.batch_size = cfg.BATCH_SIZE  #batch_size
        self.alpha = cfg.ALPHA   #leaky_relu中的alpha 

        self.offset = np.transpose(np.reshape(np.array( #将2X7X7的三维矩阵,转为7X7X2的三维矩阵
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')

        #tensor[7*7*30]
        self.logits = self.build_network( #输出logits值（预测值）
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(  #为label（真实值）穿件占位符
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class]) #[loc+conf+class]
            self.loss_layer(self.logits, self.labels)    #求loss
            self.total_loss = tf.losses.get_total_loss() #求所有的loss
            tf.summary.scalar('total_loss', self.total_loss)

    #建立网络（卷积层+池化层+全连接层）
    def build_network(self,
                      images,       #输入的图像 [None，448,448,3]
                      num_outputs,  #输出特征维度[None,7X7X30]
                      alpha,    
                      keep_prob=0.5, #dropout的参数，每个元素被保留下来的概率
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            #参考https://www.cnblogs.com/bmsl/p/dongbin_bmsl_01.html
            with slim.arg_scope( #生成一个weight变量, 用truncated normal初始化它, 并使用l2正则化
                [slim.conv2d, slim.fully_connected],   
                activation_fn=leaky_relu(alpha),  #激活函数用的是leaky_relu
                weights_regularizer=slim.l2_regularizer(0.0005),  #权重正则化用的是l2
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)  #权重初始化用的是正态分布（0.0,0.01）
            ):
                net = tf.pad(  #为输入图像进行填充，这里只填充矩阵[None，448,448,3]中的[448,448]这2维，单张图上下左右各用0填充3行/列
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), #BatchSize维度不填充，行维度上下填充3行0，列维度左右填充3列0，channel维度不填充
                    name='pad_1')
                net = slim.conv2d(  # input=net; num_outputs=64个特征图;kernel_size:7X7; strides=2;
                    net, 64, 7, 2, padding='VALID', scope='conv_2')  # 上面已经pad了，所以选padding=VALID，即不停留在图像边缘
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')#最大池化 2X2的核结构，stride=2；输出net   224X224X64
                net = slim.conv2d(net, 192, 3, scope='conv_4')       #卷积，输出特征图192个，kernel_size:3X3;  输出net: 224X224X192
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')#最大池化 2X2， stride=2； 输出net:112X112X192    
                net = slim.conv2d(net, 128, 1, scope='conv_6')       #卷积， kernel=1X1; 输出net: 112X112X128
                net = slim.conv2d(net, 256, 3, scope='conv_7')       #卷积， kernel=3X3；输出net: 112X112X256
                net = slim.conv2d(net, 256, 1, scope='conv_8')       #卷积， kernel=1X1; 输出net: 112X112X256
                net = slim.conv2d(net, 512, 3, scope='conv_9')       #卷积， kernel=3X3；输出net: 112X112X512
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')#最大池化 2X2，stride=2; 输出net: 56x56x256
                net = slim.conv2d(net, 256, 1, scope='conv_11')      #连续4组 卷积输出特征数256和512的组合；
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')      #卷积，kernel=1X1；输出net: 56x56x512
                net = slim.conv2d(net, 1024, 3, scope='conv_20')     #卷积，kernel=3X3; 输出net: 56x56x1024
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')#最大池化 2X2，stride=2；输出net:28x28x512 
                net = slim.conv2d(net, 512, 1, scope='conv_22')      #连续两组 卷积输出特征数512和1024的组合
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')     #卷积，kernel=3X3;输出net:28X28X1024
                net = tf.pad(      #对net进行填充
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')  #batch维度不填充；28的行维度上下填充1行（值为0）；28的列维度左右填充1列（值为0），channel维度不填充；
                net = slim.conv2d(      
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')#上面已经pad了，所以选padding=VALID，kernel=3X3，stride=2,输出net:14x14x1024  
                net = slim.conv2d(net, 1024, 3, scope='conv_29')      #连续两个卷积，特征数为1024，kernel=3x3
                net = slim.conv2d(net, 1024, 3, scope='conv_30')      #输出net: 7x7x1024
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')#输出net:[batchsize,channel,28,28]
                net = slim.flatten(net, scope='flat_32')              #输出net: (1,batchsize x channel x w x h)
                net = slim.fully_connected(net, 512, scope='fc_33')   #全连接层  输出net:1x512
                net = slim.fully_connected(net, 4096, scope='fc_34')  #全连接层  输出net:1x4096
                net = slim.dropout(      #dropout层，防止过拟合
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                net = slim.fully_connected(     #全连接层，输出net:7x7x30特征
                    net, num_outputs, activation_fn=None, scope='fc_36')
        return net #网络最后输出的是一个1470 维的张量（1470 = 7*7*30）

    #计算box和groundtruth的IOU值
    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0, # x-w/2=x1(左上)
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0, # y-h/2=y1(左上)
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0, # x+w/2=x2(右下)
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],# y+h/2=y2(右下)
                                axis=-1)                                # 替换最后那个维度

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point  #计算重叠区域最左上和最右下点
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)  #重叠区域
            inter_square = intersection[..., 0] * intersection[..., 1] #重叠区域面积

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3] # box1.w * box1.h
            square2 = boxes2[..., 2] * boxes2[..., 3] # box2.w * box2.h

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0) #将IOU计算得到的值归一化到（0,1）

    #定义损失函数
    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(   #将预测结果的前20（0~19）维（表示类别）转换为相应的矩阵形式（类别向量）[batchsize,7,7,20]
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(    #将预测结果的 20 ~ 21 转换为相应的矩阵形式 (置信度) [batchsize,7,7,2]
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(     #将预测的结果剩余的维度(预测的框)转变为相应的矩阵形式（boxes 所在的位置向量）[batchsize,7,7,2,4]
                predicts[:, self.boundary2:], #这里输入的box坐标形式难道是[x_i,y_i,sqrt(w_i),sqrt(h_i)]?????????
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            # 将真实的  labels 转换为相应的矩阵形式
            response = tf.reshape(  #gt中每个cell是否有物体 [bs,7,7,1] 损失函数中的l_i^obj
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(     #gt框坐标 [bs,7,7,1,4]
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(        #gt框坐标 [bs,7,7,2,4]    由于单个cell预测boxes_per_cell个box信息，先对box进行该维度上的拼贴一份相同尺度的；后将坐标尺度归一化到整幅图
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]  #gt类别    [:,20]

            offset = tf.reshape(  #将offset维度由7x7x2 reshape成 1x7x7x2
                tf.constant(self.offset, dtype=tf.float32), 
                [1, self.cell_size, self.cell_size, self.boxes_per_cell]) 
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1]) #将offset的第一个维度拼贴为batchsize大小，即offset变为：batchsize x 7x7x2
            offset_tran = tf.transpose(offset, (0, 2, 1, 3)) #作者是否考虑非AXA情况？？如7x8

            # shape为 [4, batch_size, 7, 7, 2]
            # 求中心点坐标，并归一化，offset表示的是第几个cell,例如第2个cell的中心坐标[x_center+2,y_center+0]
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,      #（预测box的x坐标（中心点坐标）+偏移量）/ 7    
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size, #（预测box的y坐标+偏移量）/ 7
                 tf.square(predict_boxes[..., 2]),            # 对sqrt(w)求平方w
                 tf.square(predict_boxes[..., 3])], axis=-1)  # 对sqrt(h)求平方h

            #计算IoU_pred^truth，置信度
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # 找到每个cell中交并比最大的，每个框只负责预测一个目标。所以object_mask就是有目标的
            # 找出iou_predict_truth 第3维度（即box_per_cell）维度计算得到的最大值构成一个tensor
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True) #有物体标志(损失函数中l_ij^obj)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response #object_mask:表示有目标 以及 目标与gt的IOU

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(   #无物体标志(损失函数中l_ij^noobj)
                object_mask, dtype=tf.float32) - object_mask

            # 参数中加上平方根是对 w 和 h 进行开平方操作，原因在论文中有说明
            # shape为(4, batch_size, 7, 7, 2)
            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss 分类损失 sum( l_i^obj * (p_i(c) - ~p_i(c))^2 ) / N
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), #对7x7x20每个维度上预测的类别做误差平方求和后，乘以损失函数系数class_scale
                name='class_loss') * self.class_scale

            # object_loss 含有object的box的confidence预测 sum( l_i^obj*(c_i - ~c_i)^2) ) / N
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss 不含object的box的confidence预测 λnoobj*sum( l_i^noobj*(c_i - ~c_i)^2) ) / N
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss 坐标损失 #shape 为 (batch_size, 7, 7, 2, 1)
            # λcoord*{sum( l_ij^obj*[(x_i - ~x_i )^2 +(y_i - ~y_i )^2]) + sum( l_ij^obj*[(sqrt(w_i) - sqrt(~w_i) )^2 +(sqrt(h_i) - sqrt(~h_i) )^2])}
            coord_mask = tf.expand_dims(object_mask, 4) # shape 为(batch_size, 7, 7, 2, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran) #需要判断第i个cell中第j个box会否负责这个object
            coord_loss = tf.reduce_mean(  #坐标四个维度对应求差[x_i,y_i,sqrt(w_i),sqrt(h_i)]-[~x_i,~y_i,sqrt(~w_i),sqrt(~h_i)]，平方和 
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            # 将所有损失放在一起
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            # 将每个损失添加到日志记录
            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
