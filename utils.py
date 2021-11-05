import torch

from config import *

from lib import *


def make_datapath_list(phase="train"):
    rootpath = "D:/PyCharmProject/pythonProject/data/furniture-images/img/"
    target_path = osp.join(rootpath + phase + "/*/*.jpg")

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


def train_model(net, dataloader_dict, loss_function, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(dataloader_dict[phase]):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = loss_function(outputs, labels)
                    _, predicts = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(predicts == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

    torch.save(net.state_dict(), save_model_file)


def load_model(net, model_path):
    weights = torch.load(model_path)
    net.load_state_dict(weights)
    return net


if __name__ == "__main__":
    a = make_datapath_list()
    print()
