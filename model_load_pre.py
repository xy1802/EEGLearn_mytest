#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torchvision as tv
import torchvision.transforms as transforms
import torch
from PIL import Image
import scipy.io as sio
import numpy as np
from PIL import Image
import warnings
import matplotlib.pyplot as plt
from Models import BasicCNN

warnings.filterwarnings('ignore')
"""i = eval(input("输入想要识别的患者编号:（1-15）"))
j = eval(input("输入想要识别的患者脑电图编号：（1-182）"))"""
i = 15
model = BasicCNN().cuda()
# print(model)
# model = torch.load("BasicCNN.pkl")
model.load_state_dict(torch.load("BasicCNN3.pth"))
# model = model().cuda()
model.eval()
torch.no_grad()
predict = []
Images = sio.loadmat("Sample Data/images_time.mat")["img"]
Mean_Images = np.mean(Images, axis=0)
Label = (sio.loadmat("Sample Data/FeatureMat_timeWin")["features"][:, -1] - 1).astype(
    int)  # corresponding to the signal label (i.e. load levels).
Patient_id = sio.loadmat("Sample Data/trials_subNums.mat")['subjectNum'][0]  # corresponding to the patient id
image1 = Mean_Images[Patient_id == i]
for j in range(0, len(Label[Patient_id == i])):
    ###导入模型###
    """device = torch.device('cuda')
    net = BasicCNN()
    net.load_state_dict_('BasicCNN2.pkl')
    #torch.no_grad()
    net = net()
    net=net.to(device)
    torch.no_grad()"""
    """count = 0
    num = 0"""

    image = image1[j:j + 1]

    label = Label[Patient_id == i]
    """i=1
    for j in range(1,184):
        image=Mean_Images[Patient_id==i][j:(j+1)]
        label=Label[Patient_id==i]
        outputs = net(torch.tensor(image).float().cuda())
        _, predicted = torch.max(outputs.cpu().data, 1)
        num = num+1
        if predicted==label[j]:
            count = count+1

    print(count)
    print(num)
    print(count/num)
    #a = np.max(image)
    #print(a)
    #image2 = 200*image

    ###绘制二维图像####
    img1 = (Mean_Images[Patient_id==i][j]-np.min(Mean_Images[Patient_id==i][j]))
    img2 = img1/(np.max(img1))
    img3 = img2.transpose(1,2,0)
    im = Image.fromarray((img3 * 255).astype(np.uint8)).convert('RGB')
    #im.save()
    #img = Image.fromarray(im).convert('RGB')
    #im.show()
    plt.imshow(im)
    plt.show()
    """

    # img_ = image.to(device)

    outputs = model(torch.tensor(image).to(torch.float32).cuda())
    _, predicted = torch.max(outputs.cpu().data, 1)
    predict.append(predicted.item())

    # print("grandtruth:%s"%label[j])
    # print("=========")

    # 计算准确率
count = 0
for m in range(len(predict)):
    if ( predict[m]==label[m]):
            count = count+1

print(predict)
print("all the labels of this patient")
print(label)
print("总共测试样本数%d" % len(label))
print("正确的标签有%d个" % count)
print("正确率为%.2f" % (count/len(label)))