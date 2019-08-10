from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )   # 验证和测试数据集时 shuffle=False， 因为设置成True 没有意义

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []  # 用来存放 所有图片 的检测结果的标签
    sample_metrics = []  # List of tuples (TP, confs, pred)  # 记录所有图片的检测结果
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        #  targets 是多张图片 的检测结果
        #  targets [:, 0] 和1 值得是什么

        # Extract labels
        labels += targets[:, 1].tolist()  # 将所有对象的类别编号存放在列表labels中 ?????????????
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])  # 坐标转换，将YOLO中心点表示的比例坐标 转换成 两个点表示的比例坐标
        targets[:, 2:] *= img_size  # 比例坐标乘以图片的尺寸 等于 点的坐标值（计算出来的坐标值应该是小数，按理说应该是整数）

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)  # 输入 多张图片 预测结果
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)  # 对一张图片的检测结果 进行非极大抑制

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)  # 记录所有图片的 检测结果

    # Concatenate sample statistics  连接 样本 统计信息
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)  # 解析配置文件 coco.data 中的信息，返回的是字典
    valid_path = data_config["valid"]  # 验证数据集TXT  valid=data/coco/5k.txt “data/coco/5k.txt”
    class_names = load_classes(data_config["names"])  # 读取配置文件中，指定的coco name的名称列表

    # Initiate model
    model = Darknet(opt.model_def).to(device)  # 加载YOLOv3(Darknet)模型
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)  # 加载模型参数  weights/yolov3.weights
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
