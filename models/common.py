# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box, feature_visualization
from utils.torch_utils import time_synchronized

from torch.nn import init, Sequential
from torchmetrics import StructuralSimilarityIndexMeasure

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAMHS(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAMHS, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        rgb, inf = x[0], x[1]
        
        rgb = self.spatial_attention(rgb) * rgb
        inf = self.spatial_attention(inf) * inf
        cat = torch.cat((rgb, inf),dim=1)
        out = self.channel_attention(cat) * cat
        return out


class SE(nn.Module):
    def __init__(self, c1, c2, r=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])


class Add_list(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class SelfAttentionTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x

class ComplexAttention(nn.Module):
    def __init__(self, d_model):
        super(ComplexAttention, self).__init__()
        self.h = 8
        self.d_model = d_model
        self.d_k = d_model // self.h
        self.d_v = d_model // self.h
        
        self.query_conv = nn.Conv2d(d_model, self.d_k * self.h, kernel_size=1)
        self.key_conv = nn.Conv2d(d_model, self.d_k * self.h, kernel_size=1)
        self.value_conv = nn.Conv2d(d_model, self.d_v * self.h, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # regularization
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1).contiguous().view(B, H*W, self.h, self.d_k).permute(0, 2, 1, 3)#b,head,hw,dk
        proj_key = self.key_conv(x).view(B, -1, H*W).permute(0, 2, 1).contiguous().view(B, H*W, self.h, self.d_k).permute(0, 2, 3, 1)#b,head,dk, hw
        proj_value = self.value_conv(x).view(B, -1, H*W).permute(0, 2, 1).contiguous().view(B, H*W, self.h, self.d_k).permute(0, 2, 1, 3)#b,head,hw,dk

        attention_weights = F.softmax(torch.matmul(proj_query, proj_key) / np.sqrt(self.d_k), dim=-1)
        attention_weights = self.attn_drop(attention_weights)

        attention_out = torch.matmul(attention_weights, proj_value).permute(0, 2, 1, 3).contiguous().view(B, H*W, self.h * self.d_v)

        # print(f'attention{attention.shape}, proj_value{proj_value.shape}, B, C, H, W:{B, C, H, W}')
        out = attention_out.permute(0, 2, 1).contiguous().view(B, C, H, W)  # B x C x H x W
        out = self.gamma * out + x  # apply gamma scaling and add residual connection
        return out

class CrossBatchAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossBatchAttention, self).__init__()
        self.in_channels = in_channels
        self.attention = ComplexAttention(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    def forward(self, x):
        B, C, H, W = x.size()
        x_mean = x.mean(dim=0, keepdim=True).expand(B, C, H, W)  # mean of the entire batch
        att_out = self.attention(x)  # B x C x H x W
        out = self.conv(x_mean + att_out)
        return out
    

class CrossAttentionK(nn.Module):
    """
     Multi-head masked corss-attention layer
    """
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttentionK, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.mix_q = nn.Linear(2*d_model, d_model)
        self.mix_q_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        # self.channelatt = ChannelAttention(in_planes=6, ratio=2) # bs=8
        # self.channelatt = CrossBatchAttention(in_channels=d_model)

        # rgb: key, query, value projections for all heads
        # self.rgb_que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.rgb_key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.rgb_val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.rgb_out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # infrared: key, query, value projections for all heads
        # self.inf_que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.inf_key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.inf_val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.inf_out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x12, attention_mask=None, attention_weights=None):
        '''
        Computes Cross-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        x, x2 = x12[0], x12[1]
        b_s, nq, c = x.shape
        nk = x.shape[1]
        assert nq == 64

        mix_q = self.mix_q(torch.cat((x, x2),dim=-1)) # bs, nq, c
        mix_q = self.mix_q_proj(mix_q).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3) #bs, nq, h, dk -> bs, h, nq, dk

        # rgb: q k v
        rgb_k = self.rgb_key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        rgb_v = self.rgb_val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        # inf: q k v
        inf_k = self.inf_key_proj(x2).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        inf_v = self.inf_val_proj(x2).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        rgb_att = torch.matmul(mix_q, inf_k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        inf_att = torch.matmul(mix_q, rgb_k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            rgb_att = rgb_att * attention_weights
            inf_att = inf_att * attention_weights
        if attention_mask is not None:
            rgb_att = rgb_att.masked_fill(attention_mask, -np.inf)
            inf_att = inf_att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        rgb_att = torch.softmax(rgb_att, -1)
        inf_att = torch.softmax(inf_att, -1)
        rgb_att = self.attn_drop(rgb_att)
        inf_att = self.attn_drop(inf_att)

        # output
        rgb_out = torch.matmul(rgb_att, rgb_v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        inf_out = torch.matmul(inf_att, inf_v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        rgb_out = self.resid_drop(self.rgb_out_proj(rgb_out))  # (b_s, nq, d_model)
        inf_out = self.resid_drop(self.inf_out_proj(inf_out))  # (b_s, nq, d_model)

        return rgb_out, inf_out


class CrossAttentionTransformerK(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input_rgb = nn.LayerNorm(d_model)
        self.ln_input_inf = nn.LayerNorm(d_model)
        self.ln_output_rgb = nn.LayerNorm(d_model)
        self.ln_output_inf = nn.LayerNorm(d_model)
        self.corssak = CrossAttentionK(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_rgb = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.LeakyReLU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(block_exp * d_model, d_model),
        )

        self.mlp_lwir = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.LeakyReLU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(block_exp * d_model, d_model),
        )

        self.dropout1_rgb = nn.Dropout(attn_pdrop)
        self.dropout2_rgb = nn.Dropout(attn_pdrop)
        self.dropout1_lwir = nn.Dropout(attn_pdrop)
        self.dropout2_lwir = nn.Dropout(attn_pdrop)

    def forward(self, x12):
        x, x2 = x12[0], x12[1]
        bs, nx, c = x.size()

        cax, cax2 = self.corssak((x,x2))

        x = self.ln_input_rgb(x + self.dropout1_rgb(cax))
        x2 = self.ln_input_inf(x2 + self.dropout1_rgb(cax2))

        # ffn
        x = self.ln_output_rgb(x + self.dropout2_rgb(self.mlp_rgb(x)))
        x2 = self.ln_output_inf(x2 + self.dropout2_lwir(self.mlp_lwir(x2)))

        return x, x2


class CrossAttentionV(nn.Module):
    """
     Multi-head masked corss-attention layer
    """
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttentionV, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.mix_q = nn.Linear(2*d_model, d_model)
        self.mix_q_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        # self.channelatt = ChannelAttention(in_planes=6, ratio=2) # bs=8
        # self.channelatt = CrossBatchAttention(in_channels=d_model)

        # rgb: key, query, value projections for all heads
        self.rgb_key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.rgb_val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.rgb_out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # infrared: key, query, value projections for all heads
        self.inf_key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.inf_val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.inf_out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x12, attention_mask=None, attention_weights=None):
        '''
        Computes Cross-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        x, x2 = x12[0], x12[1]
        b_s, nq, c = x.shape
        nk = x.shape[1]
        assert nq == 64

        mix_q = self.mix_q(torch.cat((x, x2),dim=-1)) # bs, nq, c
        mix_q = self.mix_q_proj(mix_q).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3) #bs, nq, h, dk -> bs, h, nq, dk

        # rgb: q k v
        rgb_k = self.rgb_key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        rgb_v = self.rgb_val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        # inf: q k v
        inf_k = self.inf_key_proj(x2).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        inf_v = self.inf_val_proj(x2).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        rgb_att = torch.matmul(mix_q, rgb_k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        inf_att = torch.matmul(mix_q, inf_k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            rgb_att = rgb_att * attention_weights
            inf_att = inf_att * attention_weights
        if attention_mask is not None:
            rgb_att = rgb_att.masked_fill(attention_mask, -np.inf)
            inf_att = inf_att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        rgb_att = torch.softmax(rgb_att, -1)
        inf_att = torch.softmax(inf_att, -1)
        rgb_att = self.attn_drop(rgb_att)
        inf_att = self.attn_drop(inf_att)

        # output
        rgb_out = torch.matmul(rgb_att, inf_v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        inf_out = torch.matmul(inf_att, rgb_v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        rgb_out = self.resid_drop(self.rgb_out_proj(rgb_out))  # (b_s, nq, d_model)
        inf_out = self.resid_drop(self.inf_out_proj(inf_out))  # (b_s, nq, d_model)

        return rgb_out, inf_out


class CrossAttentionTransformerV(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input_rgb = nn.LayerNorm(d_model)
        self.ln_input_inf = nn.LayerNorm(d_model)
        self.ln_output_rgb = nn.LayerNorm(d_model)
        self.ln_output_inf = nn.LayerNorm(d_model)
        self.corssav = CrossAttentionV(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_rgb = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.LeakyReLU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(block_exp * d_model, d_model),
        )

        self.mlp_lwir = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.LeakyReLU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(block_exp * d_model, d_model),
        )

        self.dropout1_rgb = nn.Dropout(attn_pdrop)
        self.dropout2_rgb = nn.Dropout(attn_pdrop)
        self.dropout1_lwir = nn.Dropout(attn_pdrop)
        self.dropout2_lwir = nn.Dropout(attn_pdrop)

    def forward(self, x12):
        x, x2 = x12[0], x12[1]
        bs, nx, c = x.size()

        cax, cax2 = self.corssav((x,x2))

        x = self.ln_input_rgb(x + self.dropout1_rgb(cax))
        x2 = self.ln_input_inf(x2 + self.dropout1_rgb(cax2))

        # ffn
        x = self.ln_output_rgb(x + self.dropout2_rgb(self.mlp_rgb(x)))
        x2 = self.ln_output_inf(x2 + self.dropout2_lwir(self.mlp_lwir(x2)))

        return x, x2


class AttentionalPositionEncoding(nn.Module):
    def __init__(self, d_model, h, attn_pdrop=.1, resid_pdrop=.1):
        super(AttentionalPositionEncoding, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        self.cnn_module = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, bias=False)
        self.fc_pos = nn.Linear(d_model, d_model)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
    def prepare_position_code(self, x):
        # B x d_model x H x W
        B, _, H, W = x.size()
        pos_code = torch.zeros(B, self.d_model, H, W, device=x.device)
        for i in range(H):
            for j in range(W):
                pos_code[:, :, i, j] = torch.randn(B, self.d_model)
        return pos_code
    
    def forward(self, x):
        B, C, H, W = x.size()
        pos_code = self.prepare_position_code(x).view(B, self.d_model, -1).permute(0, 2, 1)# B hw, C
        cnn_out = self.cnn_module(x).view(B, self.d_model, -1).permute(0, 2, 1)# B hw, C
        # print(f'x.shape{x.shape}, cnn_out.shape{cnn_out.shape}, B, C, H, W: {B, C, H, W}')

        q = self.que_proj(cnn_out).view(B, H*W, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.key_proj(pos_code).view(B, H*W, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.val_proj(pos_code).view(B, H*W, self.h, self.d_v).permute(0, 2, 1, 3)

        attention_weights = F.softmax(torch.matmul(q, k) / np.sqrt(self.d_k), dim=-1)
        attention_weights = self.attn_drop(attention_weights)

        attention_out = torch.matmul(attention_weights, v).permute(0, 2, 1, 3).contiguous().view(B, H*W, self.h * self.d_v)

        x_view = x.view(B, self.d_model, -1).permute(0, 2, 1)
        pos_encoded = self.out_proj(self.fc_pos(cnn_out) + attention_out)
        pos_encoded = self.resid_drop(pos_encoded)
        return pos_encoded + x_view
    

class MSTF(nn.Module):
    """ 
    MSTF: MultiSpectral Transformer Fusion
    """
    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=1, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        self.pos_emb_rgb = AttentionalPositionEncoding(d_model, h=8)
        self.pos_emb_inf = AttentionalPositionEncoding(d_model, h=8)

        # transformer
        self.self_trans_rgb = nn.Sequential(*[SelfAttentionTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])
        self.self_trans_inf = nn.Sequential(*[SelfAttentionTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # corss_transformer
        self.cross_transK = nn.Sequential(*[CrossAttentionTransformerK(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])
        self.cross_transV = nn.Sequential(*[CrossAttentionTransformerV(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f_rgb = nn.LayerNorm(self.n_embd)
        self.ln_f_inf = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        self.sigmoid = nn.Sigmoid()
        self.input_proj = nn.Sequential(
                nn.Conv2d(d_model, d_model//2, kernel_size=1),
                nn.GroupNorm(32, d_model//2),
                nn.Conv2d(d_model//2, d_model, kernel_size=1),
        )
        nn.init.xavier_uniform_(self.input_proj[0].weight, gain=1)
        nn.init.constant_(self.input_proj[0].bias, 0)
        #

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea0 = x[0]  # (B, C, H, W)
        inf_fea0 = x[1]   # (B, C, H, W)
        assert rgb_fea0.shape[0] == inf_fea0.shape[0]
        bs, c, h, w = rgb_fea0.shape
        rgb_fea0_pj = self.input_proj(rgb_fea0)
        inf_fea0_pj = self.input_proj(inf_fea0)

        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea0_pj)
        inf_fea = self.avgpool(inf_fea0_pj)

        # Self
        pos_emb_rgb = self.pos_emb_rgb(rgb_fea).view(bs, c, -1).permute(0, 2, 1).contiguous() # (B, H*W, C)
        pos_emb_inf = self.pos_emb_inf(inf_fea).view(bs, c, -1).permute(0, 2, 1).contiguous() # (B, H*W, C)
        # self-att-rgb
        rgb_fea_out = self.self_trans_rgb(self.drop(pos_emb_rgb)) # (B, H*W, C)
        rgb_fea_out = self.ln_f_rgb(rgb_fea_out)  # decoder head
        # self-att-inf
        inf_fea_out = self.self_trans_inf(self.drop(pos_emb_inf)) # (B, H*W, C)
        inf_fea_out = self.ln_f_inf(inf_fea_out)  # decoder head
        
        
        # Cross K
        rgb_fea_out_4d = rgb_fea_out.permute(0,2,1).view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        inf_fea_out_4d = inf_fea_out.permute(0,2,1).view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        pos_emb_rgb = self.pos_emb_rgb(rgb_fea_out_4d).view(bs, c, -1).permute(0, 2, 1).contiguous() # (B, H*W, C)
        pos_emb_inf = self.pos_emb_inf(inf_fea_out_4d).view(bs, c, -1).permute(0, 2, 1).contiguous() # (B, H*W, C)
        # drop
        x_rgb_fea = self.drop(pos_emb_rgb)
        x_inf_fea = self.drop(pos_emb_inf)
        # cross_transK
        rgb_fea_out, inf_fea_out = self.cross_transK((x_rgb_fea, x_inf_fea))
        

        # Cross V
        rgb_fea_out_4d = rgb_fea_out.permute(0,2,1).view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        inf_fea_out_4d = inf_fea_out.permute(0,2,1).view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        pos_emb_rgb = self.pos_emb_rgb(rgb_fea_out_4d).view(bs, c, -1).permute(0, 2, 1).contiguous() # (B, H*W, C)
        pos_emb_inf = self.pos_emb_inf(inf_fea_out_4d).view(bs, c, -1).permute(0, 2, 1).contiguous() # (B, H*W, C)
        # drop
        x_rgb_fea = self.drop(pos_emb_rgb)
        x_inf_fea = self.drop(pos_emb_inf)
        # cross_transV
        rgb_fea_out, inf_fea_out = self.cross_transV((x_rgb_fea, x_inf_fea))# (B, hw, C)


        # 空间尺度的插值1-线性插值法
        rgb_fea_out = rgb_fea_out.permute(0,2,1).view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        inf_fea_out = inf_fea_out.permute(0,2,1).view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bicubic') # bilinear
        inf_fea_out = F.interpolate(inf_fea_out, size=([h, w]), mode='bicubic')

        # feature_visualization(smax_rgb, title='smax_rgb')
        # feature_visualization(smax_inf, title='smax_inf')

        rgb_out = rgb_fea0 * self.sigmoid(rgb_fea_out) + rgb_fea0
        inf_out = inf_fea0 * self.sigmoid(inf_fea_out) + inf_fea0

        prob_rgb = torch.softmax(rgb_out, dim = 1)
        prob_inf = torch.softmax(inf_out, dim = 1)
        semantic_kl = F.kl_div(prob_inf, prob_rgb, reduction='mean').abs()

        return rgb_out, inf_out, semantic_kl

