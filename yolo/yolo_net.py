import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim

"""
�ο����ӣ�
1.https://blog.csdn.net/qq_34784753/article/details/78803423
2.https://blog.csdn.net/qq1483661204/article/details/79681926
3.https://blog.csdn.net/c20081052/article/details/80260726
"""

class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES         #���
        self.num_class = len(self.classes) #����� 20
        self.image_size = cfg.IMAGE_SIZE   #ͼƬ�ߴ� 448
        self.cell_size = cfg.CELL_SIZE     #cell��С 7*7
        self.boxes_per_cell = cfg.BOXES_PER_CELL #ÿ�� cell �� bounding box ���� 2
        self.output_size = (self.cell_size * self.cell_size) *\   #���tensor��С[7*7*(20+2*5)]
            (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size  # ����cell�߶�
        self.boundary1 = self.cell_size * self.cell_size * self.num_class #���е� cell ������Ԥ�������ά��[7*7*20]
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell  #�����֮��ÿ��cell ����Ӧ�� bounding boxes ���������ܺ�[7*7*(20+2)]

        # ��ʧ�����е����lamda����
        self.object_scale = cfg.OBJECT_SCALE     # ֵΪ1����Ŀ����ڵ�ϵ��   
        self.noobject_scale = cfg.NOOBJECT_SCALE # ��noobj
        self.class_scale = cfg.CLASS_SCALE       # ֵΪ2.0�� �����ʧ������ϵ��
        self.coord_scale = cfg.COORD_SCALE       # ��coord

        self.learning_rate = cfg.LEARNING_RATE  #ѧϰ��
        self.batch_size = cfg.BATCH_SIZE  #batch_size
        self.alpha = cfg.ALPHA   #leaky_relu�е�alpha 

        self.offset = np.transpose(np.reshape(np.array( #��2X7X7����ά����,תΪ7X7X2����ά����
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')

        #tensor[7*7*30]
        self.logits = self.build_network( #���logitsֵ��Ԥ��ֵ��
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(  #Ϊlabel����ʵֵ������ռλ��
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class]) #[loc+conf+class]
            self.loss_layer(self.logits, self.labels)    #��loss
            self.total_loss = tf.losses.get_total_loss() #�����е�loss
            tf.summary.scalar('total_loss', self.total_loss)

    #�������磨�����+�ػ���+ȫ���Ӳ㣩
    def build_network(self,
                      images,       #�����ͼ�� [None��448,448,3]
                      num_outputs,  #�������ά��[None,7X7X30]
                      alpha,    
                      keep_prob=0.5, #dropout�Ĳ�����ÿ��Ԫ�ر����������ĸ���
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            #�ο�https://www.cnblogs.com/bmsl/p/dongbin_bmsl_01.html
            with slim.arg_scope( #����һ��weight����, ��truncated normal��ʼ����, ��ʹ��l2����
                [slim.conv2d, slim.fully_connected],   
                activation_fn=leaky_relu(alpha),  #������õ���leaky_relu
                weights_regularizer=slim.l2_regularizer(0.0005),  #Ȩ�������õ���l2
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)  #Ȩ�س�ʼ���õ�����̬�ֲ���0.0,0.01��
            ):
                net = tf.pad(  #Ϊ����ͼ�������䣬����ֻ������[None��448,448,3]�е�[448,448]��2ά������ͼ�������Ҹ���0���3��/��
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), #BatchSizeά�Ȳ���䣬��ά���������3��0����ά���������3��0��channelά�Ȳ����
                    name='pad_1')
                net = slim.conv2d(  # input=net; num_outputs=64������ͼ;kernel_size:7X7; strides=2;
                    net, 64, 7, 2, padding='VALID', scope='conv_2')  # �����Ѿ�pad�ˣ�����ѡpadding=VALID������ͣ����ͼ���Ե
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')#���ػ� 2X2�ĺ˽ṹ��stride=2�����net   224X224X64
                net = slim.conv2d(net, 192, 3, scope='conv_4')       #������������ͼ192����kernel_size:3X3;  ���net: 224X224X192
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')#���ػ� 2X2�� stride=2�� ���net:112X112X192    
                net = slim.conv2d(net, 128, 1, scope='conv_6')       #����� kernel=1X1; ���net: 112X112X128
                net = slim.conv2d(net, 256, 3, scope='conv_7')       #����� kernel=3X3�����net: 112X112X256
                net = slim.conv2d(net, 256, 1, scope='conv_8')       #����� kernel=1X1; ���net: 112X112X256
                net = slim.conv2d(net, 512, 3, scope='conv_9')       #����� kernel=3X3�����net: 112X112X512
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')#���ػ� 2X2��stride=2; ���net: 56x56x256
                net = slim.conv2d(net, 256, 1, scope='conv_11')      #����4�� ������������256��512����ϣ�
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')      #�����kernel=1X1�����net: 56x56x512
                net = slim.conv2d(net, 1024, 3, scope='conv_20')     #�����kernel=3X3; ���net: 56x56x1024
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')#���ػ� 2X2��stride=2�����net:28x28x512 
                net = slim.conv2d(net, 512, 1, scope='conv_22')      #�������� ������������512��1024�����
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')     #�����kernel=3X3;���net:28X28X1024
                net = tf.pad(      #��net�������
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')  #batchά�Ȳ���䣻28����ά���������1�У�ֵΪ0����28����ά���������1�У�ֵΪ0����channelά�Ȳ���䣻
                net = slim.conv2d(      
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')#�����Ѿ�pad�ˣ�����ѡpadding=VALID��kernel=3X3��stride=2,���net:14x14x1024  
                net = slim.conv2d(net, 1024, 3, scope='conv_29')      #�������������������Ϊ1024��kernel=3x3
                net = slim.conv2d(net, 1024, 3, scope='conv_30')      #���net: 7x7x1024
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')#���net:[batchsize,channel,28,28]
                net = slim.flatten(net, scope='flat_32')              #���net: (1,batchsize x channel x w x h)
                net = slim.fully_connected(net, 512, scope='fc_33')   #ȫ���Ӳ�  ���net:1x512
                net = slim.fully_connected(net, 4096, scope='fc_34')  #ȫ���Ӳ�  ���net:1x4096
                net = slim.dropout(      #dropout�㣬��ֹ�����
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                net = slim.fully_connected(     #ȫ���Ӳ㣬���net:7x7x30����
                    net, num_outputs, activation_fn=None, scope='fc_36')
        return net #��������������һ��1470 ά��������1470 = 7*7*30��

    #����box��groundtruth��IOUֵ
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
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0, # x-w/2=x1(����)
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0, # y-h/2=y1(����)
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0, # x+w/2=x2(����)
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],# y+h/2=y2(����)
                                axis=-1)                                # �滻����Ǹ�ά��

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point  #�����ص����������Ϻ������µ�
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)  #�ص�����
            inter_square = intersection[..., 0] * intersection[..., 1] #�ص��������

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3] # box1.w * box1.h
            square2 = boxes2[..., 2] * boxes2[..., 3] # box2.w * box2.h

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0) #��IOU����õ���ֵ��һ������0,1��

    #������ʧ����
    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(   #��Ԥ������ǰ20��0~19��ά����ʾ���ת��Ϊ��Ӧ�ľ�����ʽ�����������[batchsize,7,7,20]
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(    #��Ԥ������ 20 ~ 21 ת��Ϊ��Ӧ�ľ�����ʽ (���Ŷ�) [batchsize,7,7,2]
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(     #��Ԥ��Ľ��ʣ���ά��(Ԥ��Ŀ�)ת��Ϊ��Ӧ�ľ�����ʽ��boxes ���ڵ�λ��������[batchsize,7,7,2,4]
                predicts[:, self.boundary2:], #���������box������ʽ�ѵ���[x_i,y_i,sqrt(w_i),sqrt(h_i)]?????????
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            # ����ʵ��  labels ת��Ϊ��Ӧ�ľ�����ʽ
            response = tf.reshape(  #gt��ÿ��cell�Ƿ������� [bs,7,7,1] ��ʧ�����е�l_i^obj
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(     #gt������ [bs,7,7,1,4]
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(        #gt������ [bs,7,7,2,4]    ���ڵ���cellԤ��boxes_per_cell��box��Ϣ���ȶ�box���и�ά���ϵ�ƴ��һ����ͬ�߶ȵģ�������߶ȹ�һ��������ͼ
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]  #gt���    [:,20]

            offset = tf.reshape(  #��offsetά����7x7x2 reshape�� 1x7x7x2
                tf.constant(self.offset, dtype=tf.float32), 
                [1, self.cell_size, self.cell_size, self.boxes_per_cell]) 
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1]) #��offset�ĵ�һ��ά��ƴ��Ϊbatchsize��С����offset��Ϊ��batchsize x 7x7x2
            offset_tran = tf.transpose(offset, (0, 2, 1, 3)) #�����Ƿ��Ƿ�AXA���������7x8

            # shapeΪ [4, batch_size, 7, 7, 2]
            # �����ĵ����꣬����һ����offset��ʾ���ǵڼ���cell,�����2��cell����������[x_center+2,y_center+0]
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,      #��Ԥ��box��x���꣨���ĵ����꣩+ƫ������/ 7    
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size, #��Ԥ��box��y����+ƫ������/ 7
                 tf.square(predict_boxes[..., 2]),            # ��sqrt(w)��ƽ��w
                 tf.square(predict_boxes[..., 3])], axis=-1)  # ��sqrt(h)��ƽ��h

            #����IoU_pred^truth�����Ŷ�
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # �ҵ�ÿ��cell�н��������ģ�ÿ����ֻ����Ԥ��һ��Ŀ�ꡣ����object_mask������Ŀ���
            # �ҳ�iou_predict_truth ��3ά�ȣ���box_per_cell��ά�ȼ���õ������ֵ����һ��tensor
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True) #�������־(��ʧ������l_ij^obj)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response #object_mask:��ʾ��Ŀ�� �Լ� Ŀ����gt��IOU

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(   #�������־(��ʧ������l_ij^noobj)
                object_mask, dtype=tf.float32) - object_mask

            # �����м���ƽ�����Ƕ� w �� h ���п�ƽ��������ԭ������������˵��
            # shapeΪ(4, batch_size, 7, 7, 2)
            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss ������ʧ sum( l_i^obj * (p_i(c) - ~p_i(c))^2 ) / N
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), #��7x7x20ÿ��ά����Ԥ�����������ƽ����ͺ󣬳�����ʧ����ϵ��class_scale
                name='class_loss') * self.class_scale

            # object_loss ����object��box��confidenceԤ�� sum( l_i^obj*(c_i - ~c_i)^2) ) / N
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss ����object��box��confidenceԤ�� ��noobj*sum( l_i^noobj*(c_i - ~c_i)^2) ) / N
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss ������ʧ #shape Ϊ (batch_size, 7, 7, 2, 1)
            # ��coord*{sum( l_ij^obj*[(x_i - ~x_i )^2 +(y_i - ~y_i )^2]) + sum( l_ij^obj*[(sqrt(w_i) - sqrt(~w_i) )^2 +(sqrt(h_i) - sqrt(~h_i) )^2])}
            coord_mask = tf.expand_dims(object_mask, 4) # shape Ϊ(batch_size, 7, 7, 2, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran) #��Ҫ�жϵ�i��cell�е�j��box��������object
            coord_loss = tf.reduce_mean(  #�����ĸ�ά�ȶ�Ӧ���[x_i,y_i,sqrt(w_i),sqrt(h_i)]-[~x_i,~y_i,sqrt(~w_i),sqrt(~h_i)]��ƽ���� 
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            # ��������ʧ����һ��
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            # ��ÿ����ʧ��ӵ���־��¼
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
