from tensorflow.python.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import colorsys
from timeit import default_timer as timer
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
# from keras.utils import multi_gpu_model
import numpy as np
import os
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


class YOLO(object):
    def __init__(self):
        self.classes_path = 'model_data/coco_classes.txt'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.model_path = 'logs/trained_weights.h5'
        # self.gpu_num = 1
        self.model_image_size = (416, 416)
        self.score = 0.3
        self.iou = 0.45

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        # 打开一个会话
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    # 将类别放入列表class__names中
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # 得到anchor数组（9，2）
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]

        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        # 验证h5模型文件可行性
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)  # 9
        num_classes = len(self.class_names)  # 2
        is_tiny_version = num_anchors == 6  # default setting

        # 如果有异常执行except，没有异常执行else
        try:
            # 加载模型
            self.yolo_model = load_model(model_path, compile=False)
        except:
            # 可以忽略
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            # 确保最后一层输出的最后一维是3*（5+num_classes）
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 生成绘制边框的颜色，HSV为颜色模型，其中，h(色调)、s(饱和度)、v(明亮)，面向用户
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        # 将hsv转换为rgb格式，hsv取值在[0,1]之间，而rgb取值在[0,255]之间，所以要*255，面向硬件
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))

        # if self.gpu_num >= 2:
        #   self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        # 将图像转换为（416，416，3）
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # 归一化，并添加batch维度 （1，416，416，3）
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 进行预测，得到所有的anchor、得分以及所属类别的索引
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,  # 图像数据
                self.input_image_shape: [image.size[1], image.size[0]],  # 图像尺寸
                # 学习模式，0：预测模型 1：训练模型
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # 显示的字体和线宽（高宽和的1/300）
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        # 循环每一个目标
        for i, c in reversed(list(enumerate(out_classes))):
            # 提取每个目标的类别、框以及得分信息
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # 显示标签：类别和得分
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            # 显示anchor
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        # 计算图像处理时间
        end = timer()
        print(end - start)

        # 返回处理结果
        return image, out_classes, out_scores, out_boxes