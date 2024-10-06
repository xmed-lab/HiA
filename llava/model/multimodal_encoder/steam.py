from transformers import  SwinModel
import torch.nn as nn
import torch
from .resnet import ResNet50
# from resnet import ResNet50
# 

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, H=64, W=64, pre_trained = False, resnet_path = None):
        super().__init__()
        self.stem = ResNet50(4)
        # self.stem = SwinModel.from_pretrained('/home/xuhang/dingxinpeng/code/Weight/swin-base-patch4-window12-384')
        # print(resnet_path)
        # ssss
        if resnet_path is not None:
            state_dict = torch.load(resnet_path)
            print('load pre-trained stem')
            self.stem.load_state_dict(state_dict=state_dict,strict=False)
        # self.stem = nn.Sequential(*[
        #     nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.SyncBatchNorm(inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.SyncBatchNorm(inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # ])
        # self.adaptivepool = nn.AdaptiveAvgPool2d((H, W))
        # self.conv2 = nn.Sequential(*[
        #     nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(2 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv3 = nn.Sequential(*[
        #     nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(4 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv4 = nn.Sequential(*[
        #     nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(4 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4 = nn.Conv2d(2048, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Linear(768, embed_dim)
        
    def forward(self, x):
        # print(x.shape)
        c4 = self.stem(x)
        # print(c4.shape)
        # c1 = self.adaptivepool(c1)
        # c4 = c4.last_hidden_state
        # print(c4.size())
        # c4 = self.fc4(c4)
        # print(c4.shape)
        bs, dim, h, w= c4.shape
        c4 = c4.view(bs, dim, -1)
        # h4, w4 = int(l**0.5), int(l**0.5)
        # print(h4, w4)
        # # print(c4.shape)
        # # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        # # c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        # # c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        # c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
        return c4
        # return c4, c4, h4, w4


# SPM = SpatialPriorModule(resnet_path='/home/xuhang/dingxinpeng/share99/code/Weight/resnet50-0676ba61.pth')


# samples = torch.randn((1,3,672,672)) #  B,  C, H , W


# feat = SPM(samples)
# print(feat.shape)