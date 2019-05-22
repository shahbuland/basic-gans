from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np
import os
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import ezdatasets as ezd


deconv_mode = "transpose" # currently supports "transpose" and "upsample"
use_dropout = False
norm = "sigmoid" # currently supports "sigmoid" and "tanh"
load_checkpoints = False
cuda = True # Set true if you want to use gpu, false otherwise
training_data = ezd.GetTrainingData("car", norm_style=norm)
training_data = np.moveaxis(training_data,3,1)

print("data shape", training_data.shape)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

        
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.n_layers = 4

        # FC layer gen

        self.fc = nn.Linear(100,512*4*4)
        # Conv layers for generator

        # list of channels for different filters
        ch = [512,256,128,64,3]

        # 4 -> 8 -> 16 -> 32 -> 64
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(len(ch)-1):
            self.conv.append(self.myconv(ch[i],ch[i+1]))
            self.bn.append(nn.BatchNorm2d(ch[i+1]))

        self.up = nn.Upsample(scale_factor=2)

    # Custom layer for upsample+conv
    def myconv(self,fi,fo,k=4,s=1,p=1):
        if deconv_mode == "upsample":
            return nn.Conv2d(fi,fo,k,s,p)
        elif deconv_mode == "transpose":
            return nn.ConvTranspose2d(fi,fo,k,2,p)
        else:
            print("Invalid deconv mode")
            return -1


    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1,512,4,4)

        for i, (conv, bn) in enumerate(zip(self.conv,self.bn)):
            if deconv_mode == "upsample":
                x = self.up(x)
            x = conv(x)
            if i != len(self.conv) - 1:
                x = F.relu(x)
                x = bn(x)

        if norm == "sigmoid":
            return torch.sigmoid(x)
        elif norm == "tanh":
            return torch.tanh(x)
        else:
            print("Invalid norm")
            return -1


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.n_layers = 4

        # Conv layers for discriminator

        # list of channels for different filters
        ch = [3,64,128,256,512]

        # 64 -> 32 -> 16 -> 8 -> 4
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(len(ch)-1):
            self.conv.append(self.myconv(ch[i],ch[i+1]))
            self.bn.append(nn.BatchNorm2d(ch[i+1]))

        self.validity = nn.Linear(512*4*4,1)

    # Custom conv layer for my purposes
    # Takes filters in, filters out
    def myconv(self,fi,fo,k=4,s=2,p=1):
        return nn.Conv2d(fi,fo,k,s,p)

    def forward(self,x):
        for i, (conv, bn) in enumerate(zip(self.conv,self.bn)):
            x = conv(x)
            x = nn.LeakyReLU(0.2,inplace=True)(x)
            if i != len(self.conv) - 1:
                if use_dropout:
                    x = nn.Dropout2d(0.25)(x)
                x = bn(x)

        x = x.view(-1,512*4*4)
        x = self.validity(x)
        return torch.sigmoid(x)


G = Generator()
D = Discriminator()

x = torch.from_numpy(np.random.normal(0,1,size=(1,100))).float()

# Test shape of output from whole model to be sure of input/output shapes
print("pass")
print(x.shape)
print(G(x).shape)
print(D(G(x)).shape)





loss_func = nn.BCELoss()



if cuda:
    G.cuda()
    D.cuda()
    loss_func.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor


G.apply(weights_init_normal)
D.apply(weights_init_normal)


d_opt = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))
g_opt = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))


# Load checkpoints if they are there
try:
    if load_checkpoints:
        G.load_state_dict(torch.load("Gparams.pt"))
        D.load_state_dict(torch.load("Dparams.pt"))
except:
    print("No checkpoints found")





#This function draws samples using matplotlib
def ShowSamples(title):
    # show 5 x 5 samples
    fig,axs = plt.subplots(5,5)
    X = np.random.normal(0,1,size=(25,100))
    X = Tensor(X).float()
    Y = G(X)
    # tensors to numpy arrays
    Y = Y.detach().cpu().numpy()
    if(norm == "tanh"):
        Y = 0.5*Y + 0.5
    elif(norm == "sigmoid"):
        Y = 1*Y
    Y = np.moveaxis(Y,1,3)

    cnt = 0
    
    for i in range(5):
        for j in range(5):
            axs[i][j].imshow(Y[cnt])
            cnt += 1
            
    plt.savefig(title+".png")
    plt.close()


# In a normal DCGAN, the discriminator is trained to output
# a 1 when it sees a real image an a 0 when it sees a fake image
# The generator is trained to trick the discriminator into producing a
# 1 (i.e. it tries to generate an image that makes the discriminator go:
# "I outputted 1 because this looks like a real image!"
def train(iterations):


    # batch size, interval on which to draw
    # samples using matplotlib and
    # interval on which to save model
    batch_size = 8
    sample_interval = 100
    checkPoint_interval = 250

    # make labels
    valid = Tensor(batch_size,1).fill_(1.0)
    fake = Tensor(batch_size,1).fill_(0.0)
    
    G.train()
    D.train()
    for i in range(iterations):

        
        np.random.shuffle(training_data)
        batch = training_data[0:batch_size]
        batch = Tensor(batch)
        
        # forward
        noise = Variable(Tensor(
            np.random.normal(0,1,size=(batch_size,100))).float())
        gen_img = G(noise)
        gen_labels = D(gen_img)
        batch_labels = D(batch)
        
        # backward

        # train generator

        g_opt.zero_grad()
        g_loss = loss_func(gen_labels,valid)
        g_loss.backward(retain_graph=True)
        g_opt.step()
        train_loss_g = g_loss.item()
                             
        # train discriminator

        d_opt.zero_grad()
        d_loss_real = loss_func(batch_labels,valid)
        d_loss_fake = loss_func(gen_labels,fake)
        d_loss = (d_loss_real+d_loss_fake)/2
        d_loss.backward()
        d_opt.step()
        train_loss_d = d_loss.item()

        
        print(i,"G loss:",train_loss_g,"D loss:",train_loss_d)

        
        if (i+1)%sample_interval == 0:
            ShowSamples(str(i))
        if (i+1)%checkPoint_interval == 0:
            torch.save(G.state_dict(),"Gparams.pt")
            torch.save(D.state_dict(),"Dparams.pt")



train(2500)
