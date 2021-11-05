import image_transformer
from utils import *
from lib import *


class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        label = img_path.split("\\")[-2]

        if label == "bed":
            label = 0
        elif label == "chair":
            label = 1
        elif label == 'sofa':
            label = 2
        elif label == 'swivelchair':
            label = 3
        else:
            label = 4

        return img_transformed, label


if __name__ == "__main__":
    file_list = make_datapath_list('train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformer = image_transformer.ImageTransformer(224, mean, std)
    data = MyDataset(file_list, transformer, "train")
    img, label = data.__getitem__(1001)
    print()
