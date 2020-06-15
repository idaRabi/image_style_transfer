import os
from torchvision import transforms
import torchvision.datasets as datasets
import torch
import styler.mehdi_vgg as new_model
import torch.nn as nn
from torch.autograd import Variable
from styler.ImageLoader import ImageLoader
import numpy as np

NUM_WORKER = 1
BATCH_SIZE = 10
NUM_CLASSES = 400
MOMENTUM = 0.2
WEIGHT_DECAY = 0.9
LR = 0.9

model = new_model.vgg19_mehdi(NUM_CLASSES)
criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), LR,
                            momentum=MOMENTUM,
                            weight_decay=WEIGHT_DECAY)

tiny_image_location = "/Users/Momeda/PycharmProjects/deep_learning_tehran/deep_fifth_homework/tiny_image/tiny-imagenet-200"
traindir = os.path.join(tiny_image_location, 'train')
valdir = os.path.join(tiny_image_location, 'val')
testdir = os.path.join(tiny_image_location, 'test')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=(None is None),
        num_workers=NUM_WORKER, pin_memory=True, sampler=None)

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
#test_images_path = "/Users/Momeda/PycharmProjects/deep_learning_tehran/deep_fifth_homework/tiny_image/tiny-imagenet-200/test/images"
#test_loader = ImageLoader(test_images_path, None, train_dataset.class_to_idx, test_transform)

val_images_path = "/Users/Momeda/PycharmProjects/deep_learning_tehran/deep_fifth_homework/tiny_image/tiny-imagenet-200/val/images"
val_labels_path = "/Users/Momeda/PycharmProjects/deep_learning_tehran/deep_fifth_homework/tiny_image/tiny-imagenet-200/val/val_annotations.txt"
val_loader = ImageLoader(val_images_path, val_labels_path, train_dataset.class_to_idx, test_transform)




###### todo train
for epoch in range(100):
    model.train()
    losses = 0
    seen_count = 0
    for i, (input, target) in enumerate(train_loader):
        target = Variable(target)
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        losses += loss.item() * input.size(0)
        seen_count += input.size(0)
        optimizer.step()
        #if i % 10 == 0:
        print("train epoch %d %.3f loss is %.4f for %d pic" % (epoch, i / len(train_loader), losses/seen_count, seen_count))


# #####todo test
# for epoch in range(100):
#     model.eval()
#     losses = 0
#     seen_count = 0
#     for i, (input, target) in enumerate(test_loader):
#         target = Variable(target)
#         output = model(input)
#         loss = criterion(output, target)
#         optimizer.zero_grad()
#         loss.backward()
#         losses += loss.item() * input.size(0)
#         seen_count += input.size(0)
#         optimizer.step()
#         #if i % 10 == 0:
#         print("test epoch %d %.3f loss is %.4f for %d pic" % (epoch, i / len(test_loader), losses/seen_count, seen_count))

criterion = nn.CrossEntropyLoss()
#####todo validate
for epoch in range(100):
    model.eval()
    losses = 0
    seen_count = 0
    for i, (input, target) in enumerate(val_loader):
        target = [target]
        input = input.unsqueeze(0)
        # target = np.zeros((1, NUM_CLASSES))
        # target[0, target_index] = 1
        target = Variable(torch.LongTensor(target))
        #target = target.unsqueeze(0)
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        losses += loss.item() * input.size(0)
        seen_count += input.size(0)
        optimizer.step()
        #if i % 10 == 0:
        print("evaluate epoch %d %.3f loss is %.4f for %d pic" % (epoch, i / len(val_loader), losses/seen_count, seen_count))
