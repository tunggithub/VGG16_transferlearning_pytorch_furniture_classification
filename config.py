from lib import *

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

resize = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

num_epochs=2
batch_size=64
save_model_file = 'weight.pth'