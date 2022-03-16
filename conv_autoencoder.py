import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import datetime

print("Read dataloader", flush=True)
lens = 40
Data_02Nami=np.load("../NDAcquisition/Classification/imread_02Namix"+str(lens)+"_nomask.npy",allow_pickle=True)
Data_01=np.load("../NDAcquisition/Classification/imread_01x"+str(lens)+"_nomask.npy",allow_pickle=True)
dataset=[]
shape_0=84
shape_1=84
for n in range(130147):
    img = Data_01[n]
    if shape_0==img.shape[0] and img.shape[1]==shape_1:
        dataset.append(img/255)
dataset=np.array(dataset)
print("dataset.shape",dataset.shape, flush=True)
dataset = np.transpose(dataset, (0, 3, 1, 2))
print("dataset.shape",dataset.shape, flush=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

print("Autoencoder", flush=True)
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # b, 64, 42, 42
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # b, 64, 42, 42
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # b, 64, 21, 21
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # b, 32, 21, 21
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),  # b, 16, 11, 11
            nn.Tanh(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # b, 16, 6, 6
            nn.Tanh(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),  # b, 8, 2, 2
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1),  # b, 16, 4, 4
            nn.Tanh(),
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1),  # b, 32, 8, 8
            nn.Tanh(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # b, 32, 16, 16
            nn.Tanh(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # b, 32, 32, 32
            nn.Tanh(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # b, 32, 64, 64
            nn.Tanh(),
            nn.ConvTranspose2d(32, 8, 8, stride=1, padding=1),  # b, 8, 69, 69
            nn.Tanh(),
            nn.ConvTranspose2d(8, 3, 8, stride=1, padding=1),  # b, 3, 74, 74
            nn.Tanh(),
            nn.ConvTranspose2d(3, 3, 8, stride=1, padding=1),  # b, 3, 79, 79
            nn.Tanh(),
            nn.ConvTranspose2d(3, 3, 8, stride=1, padding=1),  # b, 3, 84, 84
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder().cuda()
criterion = nn.MSELoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

starttime = datetime.datetime.now()



print("Start train", flush=True)
if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

def to_img(x):
    x = x.view(x.size(0), 3, 84, 84)
    return x

num_epochs = 10000
for epoch in range(num_epochs):
    for img in dataloader:
        img = Variable(img.float()).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(img)
        loss = criterion(output, img)
        # backward
        loss.backward()
        optimizer.step()
        
    # ===================log========================
    endtime = datetime.datetime.now()
    if epoch % 100 == 0: 
        print('>>> epoch [{}/{}], loss:{:.4f}, time:{:.2f}s'.format(epoch+1, num_epochs, loss.item(), (endtime-starttime).seconds))
        pic = to_img(output.cpu().data)
        save_image(pic, './vae_img/image_{}.png'.format(epoch))
        

torch.save(model.state_dict(), './conv_autoencoder.pth')