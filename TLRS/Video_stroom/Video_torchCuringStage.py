# coding=utf-8
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torchvision
import os
import json
import cv2
from model.model_torch import efficientnetv2_s
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# 使得视频流显示中文
def cv2ImgAddText(img, text):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fillColor = (255, 0, 0)
    fontStyle = ImageFont.truetype("font/simsun.ttc", 40, encoding='utf-8')
    draw.text((0, 0), text, font=fontStyle, fill=fillColor)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式


def Viseo_torchPredict():

    #使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载模型
    # model = torchvision.models.resnet18(pretrained=True)
    model = efficientnetv2_s(num_classes=10)
    # model_weight_path = "./efficientnetv2_s_F_best_network.pth"
    model_weight_path = r"./weight/efficientnetv2_s_MBT_best_network.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 加载标签
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    # with open(json_path, "r") as f:
    #     labels_dict = json.load(f)
    with open('./weight/CuringStage_Tobacco.txt', "r", encoding='utf-8') as f:
        labels_dict = eval(f.read())
    labels = [labels_dict[i] for i in range(len(labels_dict))]

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    frame_count = 1

    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        # 将图像转换为 PyTorch 张量
        img_size = {"s": [300, 384],  # train_size, val_size
                    "m": [384, 480],
                    "l": [384, 480]}
        num_model = "s"

        data_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(img_size[num_model][1]),
             transforms.CenterCrop(img_size[num_model][1]),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        tensor = data_transform(frame)
        # expand batch dimension
        tensor = torch.unsqueeze(tensor, dim=0)

        # 使用模型进行预测
        outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)
        label = labels[predicted[0]]

        frame = cv2ImgAddText(frame, label)
        cv2.imshow("TLFS Recognition System", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# Viseo_torchPredict()