from models.senet import senet50
import utils
from torch.autograd import Variable
import torch
import cv2
import torchvision.transforms
import numpy as np
from PIL import Image


class VGGFace:

    def __init__(self, trained_model="/media/haoxue/WD/VGGFace2-pytorch/senet50_ft_weight.pkl", transform=True):

        self.net = senet50(num_classes=8631, include_top=False)
        utils.load_state_dict(self.net, trained_model)
        self.net.eval()
        self.transform = transform

    def process(self, img_path):

        out = self.net(self.load_image(img_path))
        output = out.view(out.size(0), -1)
        output = output.data.cpu().numpy()
        print(np.shape(output))
        return output

    def load_image(self, img_path):
        # img = Image.open(img_path)
        # img = torchvision.transforms.CenterCrop(224)(img)
        # img = np.array(img, dtype=np.uint8)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        if self.transform:
            img = self.transform_img(img)

        return Variable(img)

    @staticmethod
    def transform_img(img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= np.array([91.4953, 103.8827, 131.0912])
        img = img.transpose(2, 0, 1)  # C x H x W
        img = np.expand_dims(img, axis=0)  # 1 x C x H x W
        img = torch.from_numpy(img).float()
        return img


if __name__ == "__main__":

    v = VGGFace()
    o = v.process("/home/haoxue/Downloads/download.jpeg")

