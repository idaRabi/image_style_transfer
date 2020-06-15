import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

class VGG19(nn.Module):

    def __init__(self, content_image, style_image, content_weight, style_weight, reduced_vgg=None):
        super(VGG19, self).__init__()
        if reduced_vgg is None:
            vgg_features = models.vgg19(pretrained=True).features
        else:
            vgg_features = reduced_vgg
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_losses = []
        self.style_losses = []
        self._gram_cal = GramMat()
        module_style_feature = style_image
        module_content_feature = content_image
        add_style_layer = [1, 2]
        add_content_layer = [1]

        for x in range(4):
            self.slice1.add_module(str(x), vgg_features[x])
            #vgg_features[x].requires_grad = False
            module_style_feature = vgg_features[x](module_style_feature)
            if 1 in add_style_layer:
                style_loss = self.create_style_loss(module_style_feature)
                self.slice1.add_module("l%d_s%d" % (1, x), style_loss)
                self.style_losses.append(style_loss)

            module_content_feature = vgg_features[x](module_content_feature)
            if 1 in add_content_layer and x == 3:
                content_loss = self.create_content_loss(module_content_feature)
                self.slice1.add_module("l%d_c%d" % (1, x), content_loss)
                self.content_losses.append(content_loss)

        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_features[x])
            #vgg_features[x].requires_grad = False
            module_style_feature = vgg_features[x](module_style_feature)
            if 2 in add_style_layer:
                style_loss = self.create_style_loss(module_style_feature)
                self.slice2.add_module("l%d_s%d" % (2, x), style_loss)
                self.style_losses.append(style_loss)

            module_content_feature = vgg_features[x](module_content_feature)
            if 2 in add_content_layer and x == 8:
                content_loss = self.create_content_loss(module_content_feature)
                self.slice2.add_module("l%d_c%d" % (2, x), content_loss)
                self.content_losses.append(content_loss)

        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_features[x])
            #vgg_features[x].requires_grad = False
            module_style_feature = vgg_features[x](module_style_feature)
            if 3 in add_style_layer:
                style_loss = self.create_style_loss(module_style_feature)
                self.slice3.add_module("l%d_s%d" % (3, x), style_loss)
                self.style_losses.append(style_loss)

            module_content_feature = vgg_features[x](module_content_feature)
            if 3 in add_content_layer and x == 17:
                content_loss = self.create_content_loss(module_content_feature)
                self.slice3.add_module("l%d_c%d" % (3, x), content_loss)
                self.content_losses.append(content_loss)

        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_features[x])
            #vgg_features[x].requires_grad = False
            module_style_feature = vgg_features[x](module_style_feature)
            if 4 in add_style_layer:
                style_loss = self.create_style_loss(module_style_feature)
                self.slice4.add_module("l%d_s%d" % (4, x), style_loss)
                self.style_losses.append(style_loss)

            module_content_feature = vgg_features[x](module_content_feature)
            if 4 in add_content_layer and x == 26:
                content_loss = self.create_content_loss(module_content_feature)
                self.slice4.add_module("l%d_c%d" % (4, x), content_loss)
                self.content_losses.append(content_loss)

        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_features[x])
            #vgg_features[x].requires_grad = False
            module_style_feature = vgg_features[x](module_style_feature)
            if 5 in add_style_layer:
                style_loss = self.create_style_loss(module_style_feature)
                self.slice4.add_module("l%d_s%d" % (4, x), style_loss)
                self.style_losses.append(style_loss)

            module_content_feature = vgg_features[x](module_content_feature)
            if 5 in add_content_layer and x == 35:
                content_loss = self.create_content_loss(module_content_feature)
                self.slice5.add_module("l%d_c%d" % (5, x), content_loss)
                self.content_losses.append(content_loss)

    def create_content_loss(self, model_content_feature):
        content_feature = model_content_feature.clone()
        content_loss = ContentLoss(content_feature, self.content_weight)
        return content_loss

    def create_style_loss(self, model_style_feature):
        style_feature = model_style_feature.clone()
        style_feature_gram = self._gram_cal(style_feature)
        style_loss = StyleLoss(style_feature_gram, self.style_weight)
        return style_loss


    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        vgg_outputs = namedtuple('VggOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = vgg_outputs(h1, h2, h3, h4, h5)
        return out


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.loss_cal = nn.MSELoss()

    def forward(self, x):
        self.loss = self.loss_cal(x * self.weight, self.target)
        self.output = x
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


class GramMat(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.loss_cal = nn.MSELoss()
        self.gram_cal = GramMat()

    #todo gram matrix tu forwarde na joda
    def forward(self, x):
        self.output = x.clone()
        self.G = self.gram_cal(x)
        self.G.mul_(self.weight)
        self.loss = self.loss_cal(self.G, self.target)
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss