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
from model.ShuffleNet_V1 import ShuffleNet
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
    model = ShuffleNet(num_classes=18)
    model_weight_path = "./weight/shufflev1_TobaccoTotalData.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 加载标签
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    # with open(json_path, "r") as f:
    #     labels_dict = json.load(f)
    with open('./weight/FlueCured_Tobacco.txt', "r", encoding='utf-8') as f:
        labels_dict = eval(f.read())
    labels = [labels_dict[i] for i in range(len(labels_dict))]

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    frame_count = 1

    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        # 将图像转换为 PyTorch 张量
        data_transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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
