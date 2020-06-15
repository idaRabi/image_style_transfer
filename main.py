from pathlib import Path
from styler.vgg import VGG19
import styler.utils as utils
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

from collections import defaultdict
before = defaultdict(int)
after = defaultdict(int)

style_file_location = "/Users/Momeda/PycharmProjects/deep_learning_tehran/deep_fifth_homework/resources/style_1.jpg"
content_file_location = "/Users/Momeda/PycharmProjects/deep_learning_tehran/deep_fifth_homework/resources/content_1.jpg"

if not Path(style_file_location).is_file():
    raise FileExistsError("style file does not exist")

if not Path(content_file_location).is_file():
    raise FileExistsError("content file does not exist")

IMAGE_SIZE = 128
CONTENT_WEIGHT = 10 ** 0
STYLE_WEIGHT = 10 ** 4

style_image = utils.load_image(style_file_location, size=IMAGE_SIZE)
content_image = utils.load_image(content_file_location, size=IMAGE_SIZE)
input_image = Variable(torch.randn(content_image.data.size())).type(torch.FloatTensor)

input_param = nn.Parameter(input_image.data)
optimizer = optim.LBFGS([input_param])
content_style_model = VGG19(content_image, style_image, CONTENT_WEIGHT, STYLE_WEIGHT)
best_loss = [1000000]
best_result = [None]
run = [0]
for epoch in range(20):
    def closure():

        input_param.data.clamp_(0, 1)
        optimizer.zero_grad()
        content_style_model(input_param)
        style_score = 0
        content_score = 0

        for content_loss in content_style_model.content_losses:
            clb = content_loss.backward()
            content_score += clb.data[0]
        for style_loss in content_style_model.style_losses:
            slb = style_loss.backward()
            style_score += slb.data[0]

        loss = style_score + content_score# / len(content_style_model.style_losses) + content_score / len(content_style_model.content_losses)
        if loss < best_loss[0]:
            best_result[0] = copy.deepcopy(input_param)
            best_loss[0] = loss
        print("epoch %d" % epoch)
        print("content loss %.4f" % content_score)
        print("style loss %.4f" % style_score)

        return loss
    optimizer.step(closure)
    best = best_result[0]
    best.data.clamp_(0, 1)
    output = best.data[0]
    utils.image_show(output, IMAGE_SIZE)

best = best_result[0]
best.data.clamp_(0, 1)
output = best.data[0]
plt.figure()
utils.image_show(output, IMAGE_SIZE)
plt.ioff()
plt.show()