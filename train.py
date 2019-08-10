from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")  # 断点续训
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")  # threads 线程
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="interval between saving model weights")  # 设置每间隔多少个epochs，保存一次参数。interval n. 间隔；幕间休息；间距  、
    parser.add_argument("--evaluation_interval", type=int, default=1,
                        help="interval evaluations on validation set")  # 每间隔多少个epochs，就正在验证数据集上的评估间隔 ？？？？？？？？？？
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")  # ？？？？？？？？？？？？？？TensorFlow的

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断设备是否有 GPU

    os.makedirs("output", exist_ok=True)  # 不存在文件夹就创建，存在就不创建
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)  # 将文件内容（各种数据）解析成字典  config/coco.data
    train_path = data_config["train"]  # 访问字典内容  train=data/coco/trainvalno5k.txt
    valid_path = data_config["valid"]  # 访问字典内容 valid=data/coco/5k.txt
    class_names = load_classes(data_config["names"])  # 将文件内容（名称）解析成列表（存储80个种类的列表） names=data/coco.names

    # Initiate model 和 模型的参数
    model = Darknet(opt.model_def).to(
        device)  # default="config/yolov3.cfg"  模型初始化，实例化Darknet类 成模型model(不是之前理解的将返回值传递给model)
    model.apply(weights_init_normal)  # 初始化模型的参数 model.apply（）传递进去的weights_init_normal是一个函数，

    # If specified we start from checkpoint  如果指定，我们从检查点开始
    if opt.pretrained_weights:  # 断点续训
        if opt.pretrained_weights.endswith(".pth"):  # ".pth " 是什么文件.
            model.load_state_dict(torch.load(opt.pretrained_weights))  # 因为之前保存的只是 网络的参数（这个load_state_dict方法，是troch的方法）
        else:
            model.load_darknet_weights(opt.pretrained_weights)  # 因为之前保存的只是 网络和参数（这个load_darknet_weights方法，是作者自己写的）

    # Get dataloader
    dataset = ListDataset(train_path, augment=True,
                          multiscale=opt.multiscale_training)  # train=data/coco/trainvalno5k.txt
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,  # 自定义需要数据的返回格式。collate_fn 是一个函数。
    )

    optimizer = torch.optim.Adam(model.parameters())  # 指定 优化器（梯度下降的方法）

    metrics = [  # 指标
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):  # epochs 中的每一次都遍历所有的数据集（每一个epoch）
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):  # 将一个 epochs(dataloader) 分成多个 batch(每一个batch)
            batches_done = len(dataloader) * epoch + batch_i  # batches_done 相当于是给所有 batch的编号（绝对编号）

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            # model.forward(imgs, targets) 这里其实调用的方法就是model.forward,但为什么很多程序中毒不写成.forward呢 ????????
            loss.backward()

            if batches_done % opt.gradient_accumulations:  # 每 opt.gradient_accumulations 个 batch 就反向计算一次 参数
                # Accumulates gradient before each step  在每个步骤之前 累积梯度
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress  记录 进度
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
            epoch, opt.epochs, batch_i, len(dataloader))  # 显示遍历 epoch 和 batch 的进度

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            # [['Metrics', 'YOLO Layer 0', 'YOLO Layer 1', 'YOLO Layer 2', 'YOLO Layer 3', 'YOLO Layer 4' ……]]

            # Log metrics at each YOLO layer   # ??????????????????????  完全看不懂 yolo.metrics  model.yolo_layers  这些东西
            for i, metric in enumerate(metrics):  # 指标
                formats = {m: "%.6f" for m in metrics}  # 为 metrics 中每个指标来（metric）指定具体的格式
                formats["grid_size"] = "%2d"  # 添加两个额外的指标 和 格式
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in
                               model.yolo_layers]  # yolo.metrics 访问的是yolo的属性值，（.可以访问类的方法，也可以是属性值）
                # 以 formats[metric] 的格式，显示 yolo.metrics.get(metric, 0) 的内容

                metric_table += [[metric, *row_metrics]]  # 往 metric_table 添加新的内容

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)  # logger 是使用TensorFlow 创建的

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch  确定剩下 epoch 的大致时间
            epoch_batches_left = len(dataloader) - (batch_i + 1)  # 剩下的 epoch的数量
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)  # ？？？？？？？？？？一直不知道这个参数是什么意义

        if epoch % opt.evaluation_interval == 0:  # 每隔 opt.evaluation_interval 个 epoch 就验证一次
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set  评估验证集上的模型
            precision, recall, AP, f1, ap_class = evaluate(  # 设置验证时的参数，并计算
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:  # 保存 训练过程中的参数
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
