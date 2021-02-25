"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, LeakyReLU, BatchNormalization
# from tensorflow.keras.layers.advanced_activations import LeakyReLU
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# DBL: conv2d + bn + leakyrelu
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# resn
def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x, y])
    return x


# backbone
def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


# 生成整个yolo_v3网络结构，输入：inputs=(None, None, 3), 输出：[y1, y2, y3]
# num_anchors和num_classes用于计算y1y2y3输出维度 = (s*s*num_anchors*(5+num_classes))
def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)  # 3
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])  # reshape ->(1,1,1,3,2)

    grid_shape = K.shape(feats)[1:3]  # height, width (13, 13)
    # grid_y和grid_x用于生成网格grid，通过arange、reshape、tile的组合， 创建y轴的0~12的组合grid_y，再创建x轴的0~12的组合grid_x，
    # 将两者拼接concatenate，就是grid
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])  # shape：[grid_shape[0],grid_shape[1],1,1]

    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])  # shape：[grid_shape[1],grid_shape[0],1,1]

    # 例：如果是最后一层13x13，则构成[13,13,1,2]的栅格网络，保存每个网格的坐标从(0,0)~(13,13)
    grid = K.concatenate([grid_x, grid_y])  # shape:[13,13,1,2]
    # cast函数是类型转换函数，将目标转为dtype类型
    grid = K.cast(grid, K.dtype(feats))
    # 将feats的最后一维展开，将anchors与其他数据（类别数+4个框值+框置信度）分离, (num_box, 13, 13, 3, 80+5)
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    # xywh的计算公式，tx、ty、tw和th是feats值，而bx、by、bw和bh是输出值
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    # yolo_head():box_xy是box的中心坐标，(0~1)相对位置；box_wh是box的宽高，(0~1)相对值；
    # box_confidence是框中物体置信度；box_class_probs是类别置信度
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)

    # 将box_xy和box_wh的(0~1)相对值，转换为真实坐标，输出boxes是(y_min,x_min,y_max,x_max)的值
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])

    # 框的得分=框的置信度*类别置信度
    box_scores = box_confidence * box_class_probs

    # reshape,将框的得分展平，变为(?,80); ?:框的数目
    box_scores = K.reshape(box_scores, [-1, num_classes])

    # boxes ==> (num_boxes, 4), box_scores ==> (num_box, num_class)
    return boxes, box_scores


def yolo_eval(yolo_outputs,  # 模型输出，格式如下【（?，13,13,255）（?，26,26,255）（?,52,52,255）】
              anchors,
              num_classes,
              image_shape,  # placeholder类型的TF参数，默认(416, 416)
              max_boxes=20,  # 每张图每类最多检测到20个框
              score_threshold=.6,  # 框置信度阈值，小于阈值的框被删除
              iou_threshold=.5):  # 同类别框的IoU阈值，大于阈值的重叠框被删除
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    # 每层分配3个anchor box.如13*13分配到【6,7,8】即【（116,90）（156,198）（373,326）】
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    # 输入shape(?, 13, 13, 255);即第一维和第二维分别 * 32  ->13 * 32 = 416; input_shape: (416, 416)
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    print(input_shape)

    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)  # K.concatenate:将数据展平 ->(?,4)
    box_scores = K.concatenate(box_scores, axis=0)  # ->(?,)

    mask = box_scores >= score_threshold  # #MASK掩码，过滤小于score阈值的值，只保留大于阈值的值
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')  # #最大检测框数20
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.

        # 清理达不到置信度*类别概率阈值的的类别，第一次清理
        # 即清理所有gird预测的box中，没有物体的，概率特别小的box
        class_boxes = tf.boolean_mask(boxes, mask[:, c])  # 通过掩码MASK和类别C筛选框boxes

        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])  # 通过掩码MASK和类别C筛选框scores
        # 运行非极大抑制
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        # K.gather:根据索引nms_index选择class_boxes
        class_boxes = K.gather(class_boxes, nms_index)
        # 根据索引nms_index选择class_box_score
        class_box_scores = K.gather(class_box_scores, nms_index)

        # 把类型变成一个整数而非'00...1...000'的形式
        classes = K.ones_like(class_box_scores, 'int32') * c  # 计算类的框得分

        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    # 检测类别序号是否小于类别数，避免异常数据
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    # 预设anchor box的掩码anchor_mask，第1层678，第2层345，第3层012，倒序排列
    num_layers = len(anchors)//3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')  # 真值框，左上和右下2个坐标值和1个类别 (batch_size, 20, 5)
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # xy是box的中心点，结构是(16, 20, 2)
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # wh是box的宽和高，结构也是(16, 20, 2)
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]  # 第0和1位设置为xy，除以416，归一化
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]  # 第2和3位设置为wh，除以416，归一化
    # 此时true_boxes = (16, 20, 5)  其中最后一维：(x,y,w,h,cls)

    # 设置y_true的初始值，0填充
    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]  # [[13,13], [26,26], [52,52]]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes),
                dtype='float32') for l in range(num_layers)]  # [(16,13,13,3,7), (16,26,26,3,7), (16,52,52,3,7)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # (1,9,2)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0  # 将boxes_wh中宽w大于0的位，设为True，即含有box，结构是(16,20)

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]  # 只选择存在标注框的wh,例如：wh的shape是(7,2)
        if len(wh) == 0:  # 当前图片没有标注框
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)  # (7,1,2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算标注框box与anchor box的iou值
        # note: 注意不是每个像素都产生9个anchor，只有包含box中心的网格才产生9个anchor
        intersect_mins = np.maximum(box_mins, anchor_mins)  # box_mins(7,1,2)，anchor_mins(1,9,2)，intersect_mins(7,9,2)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)  # (7,9,2)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)  # (7,9,2)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # (7,9)
        box_area = wh[..., 0] * wh[..., 1]  # (7,1)
        anchor_area = anchors[..., 0] * anchors[..., 1]  # (1,9)
        iou = intersect_area / (box_area + anchor_area - intersect_area)  # (7,9)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)  # (7,)

        # 设置y_true的值
        for t, n in enumerate(best_anchor):  # t是box的序号，n是最优anchor的序号
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # grid_shapes是3个检测图的尺寸，将归一化的值，与框长宽相乘，恢复为具体值：13，26，52
                    i = np.floor(true_boxes[b, t, 0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)  # anchor box中的序号
                    c = true_boxes[b, t, 4].astype('int32')  # c是类别
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]  # 将xy和wh放入y_true中
                    y_true[l][b, j, i, k, 4] = 1  # 将y_true的第4位框的置信度设为1，即这张图片这个anchorbox中含有对象
                    y_true[l][b, j, i, k, 5+c] = 1  # 将y_true第5~n位的类别设为1

    # y_true的第0和1位是中心点xy，范围是(0~1)，第2和3位是宽高wh，范围是(0~1)，第4位是置信度1或0，第5~n位是类别为1其余为0
    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b, ..., 0])  # b是第几张图，将置信率为0的其他参数清0
            iou = box_iou(pred_box[b], true_box)  # 单张图片单个尺度算iou
            best_iou = K.max(iou, axis=-1)  # 先取每个grid最大的iou
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))  # 删掉小于阈值的BBOX
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        # x,y交叉熵损失，首先要置信度不为0
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)

        # 宽高损失，损失函数为总方误差
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[..., 2:4])

        # 置信度损失，交叉熵，这里没有物体的部分也要计算损失
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask

        # 分类的损失
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        # 计算一个batch的总损失
        xy_loss = K.sum(xy_loss) / mf  # mf：batch_size
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
