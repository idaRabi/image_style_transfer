from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

def load_image(image_loc, size=None):

    image_transform = transforms.Compose([
        transforms.Scale(size=size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_loc)
    if size is not None:
        image = image.resize((size, size), Image.ANTIALIAS)
    image = image_transform(image)
    image = Variable(image)
    image = image.unsqueeze(0)
    return image

plt.ion()
to_pil = transforms.ToPILImage()


def image_show(tensor_image, size):
    image = tensor_image.clone().cpu()
    image = image.view(3, size, size)  # remove the fake batch dimension
    image = to_pil(image)
    plt.imshow(image)
    # if title is not None:
    #     plt.title(title)
    plt.pause(0.001)

