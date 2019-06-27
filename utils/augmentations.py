import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])   # 按照给定维度翻转图片（水平反向翻转，镜像翻转），易知，因检测对象的特点，
    # 有的图片是不能水平翻转的。例如左转弯表示，翻转之后就变右转弯了。
    targets[:, 2] = 1 - targets[:, 2]   # 2标签框的中心点离左侧距离/图片的宽度。
    # 是图片的标签中的0是置信度，1234分别是标签的位置的高度宽度（相对值，比例），图片的高度宽度（相对值，比例）。
    # 将图片水平翻转之后，标签中应该修正的值，只有2。其他的都不需要改变。
    return images, targets
