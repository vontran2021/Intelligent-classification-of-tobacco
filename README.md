# 深度学习在烟草中的应用分享
## 前言  
该项目是对本人研究生期间的研究内容进行整理总结，总结的同时也希望能够帮助更多的小伙伴。后期如果有学习到新的知识也会与大家一起分享。  
该项目分享的主要有三点：  
1）GUI。该分享是一套完整的烟叶识别系统软件开发项目.  
2）深度学习应用。该分享是一套基于卷积神经网络的识别系统开发，涉及烟叶成熟度，烟叶烘烤阶段和烤后烟等级的识别.    
3）图像处理。该分享涉及了大量的图像处理操作代码.  
## Image  
存储系统Logo，背景  
## ImageProcessPy  
储存了：去雾，加噪声，提取图像RGB通道值图像处理算法。在子文件夹img中将会存储你运行代码的结果  
## model  
识别程序所使用的神经网络框架都在这里  
## RecognitionPy
识别程序都在这里  
## Video_stroom  
调用的实时识别代码都在这里  
## TLRS
这是识别系统的主程序，内部包含了主界面设计代码、登录界面代码、功能调用代码和部分图像处理算法
## 所需环境  
Anaconda3  
python3.6/3.7/3.8  
pycharm (IDE)  
pytorch 1.10 (pip package)  
torchvision 0.11.1 (pip package)  
torchaudio 0.10.0(pip package)  
pyqt5 5.15 (pip package)   

