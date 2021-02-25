"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


# 使用Python的Lambda表达式，顺次执行函数列表，且前一个函数的输出是后一个函数的输入
def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])  # (n, 5)

    # 如果非随机
    if not random:
        # 将图片大小转换为(416, 416, 3)
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        # 计算填充部分
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)  # (416, 208)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))  # (416, 416)
            new_image.paste(image, (dx, dy))  # 用灰色填充
            image_data = np.array(new_image)/255.  # 归一化

        # correct boxes
        box_data = np.zeros((max_boxes, 5))  # (20, 5)
        if len(box) > 0:
            np.random.shuffle(box)
            # 取最多20个框
            if len(box) > max_boxes:
                box = box[:max_boxes]
            # box按相同比例缩放
            box[:, [0, 2]] = box[:, [0, 2]]*scale + dx
            box[:, [1, 3]] = box[:, [1, 3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # 如果是随机，图片要经过一下随机增强：非等比例变换——>转换为(416,416)——>随机翻转——>色域随机变化和归一化——>image_data
    # 通过jitter参数，随机计算new_ar和scale，生成新的nh和nw，将原始图像随机转换为nw和nh尺寸的图像，即非等比例变换图像
    new_ar = w/h * rand(1-jitter, 1+jitter)/rand(1-jitter, 1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    # 将变换后的图像，转换为416x416的图像，其余部分用灰色值填充
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    # 根据随机数flip，随机左右翻转FLIP_LEFT_RIGHT图片
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    # 在HSV坐标域中，改变图片的颜色范围，hue值相加，sat和vat相乘，先由RGB转为HSV，再由HSV转为RGB，添加若干错误判断，避免范围过大。
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1/rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1 (416,416,3)

    # correct boxes
    # 将所有的图片变换，增加至检测框中，并且包含若干异常处理，避免变换之后的值过大或过小，去除异常的box
    box_data = np.zeros((max_boxes,5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
        box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box  # (20, 5)

    return image_data, box_data
