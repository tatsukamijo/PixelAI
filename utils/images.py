import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import torch


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imsave_numpy(img, path):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    # print(npimg.shape)
    plt.imshow(npimg)
    plt.savefig(path)


def imsave_pil(img, path):
    transforms.ToPILImage(mode="RGB")(img).save(path)


def imsave_torch(img, path):
    torchvision.utils.save_image(img, path)


class SubtractTactileBG(object):
    def __init__(self, bg_path):
        self.bg = transforms.ToTensor()(Image.open(bg_path))

    def __call__(self, img):
        diff = torch.abs(img - self.bg)
        # print(img[0][0])
        # print(self.bg[0][0])
        # print(diff[0][0])
        return diff


class AdjustBrightnessAndContrast(object):
    def __init__(self, brightness_factor=1, contrast_factor=1):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        return F.adjust_brightness(
            F.adjust_contrast(
                img,
                contrast_factor=self.contrast_factor,
            ),
            brightness_factor=self.brightness_factor,
        )


class MinMaxNormalize(object):
    def __call__(self, img):
        # Normalize image to [0, 1]
        img = img - torch.min(img)
        img = img / torch.max(img)
        return img


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = img * self.std + self.mean
        return img