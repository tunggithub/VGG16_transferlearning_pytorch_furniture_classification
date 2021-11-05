from dataset import *
from image_transformer import *
from utils import *
from lib import *
from config import *

if __name__ == '__main__':
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=5)

    # setting loss function

    loss_function = nn.CrossEntropyLoss()

    params_to_update = []

    update_params_name = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in net.named_parameters():
        if name in update_params_name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    # setting optimize

    optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

    transformer = ImageTransformer(224, mean, std)

    train_list = make_datapath_list('train')
    val_list = make_datapath_list('val')
    train_dataset = MyDataset(train_list, transformer, "train")
    val_dataset = MyDataset(val_list, transformer, "val")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size)
    dataloader_dict = {
        'train': train_dataloader,
        'val': val_dataloader
    }

    train_model(net, dataloader_dict, loss_function, optimizer, num_epochs)
