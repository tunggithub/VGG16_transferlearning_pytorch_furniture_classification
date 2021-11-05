import numpy as np

from lib import *
from config import *
from utils import *
from image_transformer import *


class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict(self, lastlayer):
        max_id = np.argmax(lastlayer.detach().numpy())
        return self.class_index[max_id]


class_index = ['bed', 'chair','sofa','swivelchair','table']

predictor = Predictor(class_index)


def predict(img_path):
    img = Image.open(img_path)
    # load network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=5)
    net.eval()

    # load model

    model= load_model(net, model_path='weight.pth')

    # prepare image input
    transfomer = ImageTransformer(resize, mean, std)
    img = transfomer(img, 'test')
    img = img.unsqueeze_(0)  # lam gia batch size

    # predict

    lastlayer = model(img)
    output = predictor.predict(lastlayer)

    return output


print(predict('bed.jpg'))
