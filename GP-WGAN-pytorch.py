from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np
import os
from PIL import Image
from torch import autograd
import matplotlib.pyplot as plt
import ezdatasets as ezd
from torchsummary import summary


deconv_mode = "upsample" # currently supports "transpose" and "upsample"
use_dropout = True
norm = "sigmoid" # currently supports "sigmoid" and "tanh"
load_checkpoints = False
cuda = True # True for GPU, false for CPU
n_critic = 5 # ratio on which to train discriminator
gp_weight = 10 # Weight given to GP Loss
batch_size = 128
sample_interval = 100
checkPoint_interval = 250
channels = 3
img_size = 64
iters = 2500

MAX_FILTERS = 256 # The largest number of filters used in CNNs

training_data = ezd.GetTrainingData("person", shape=(img_size,img_size),
                                    norm_style=norm)
training_data = np.moveaxis(training_data,3,1)

print("data shape:", training_data.shape)


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

        # FC layer gen

        self.fc = nn.Linear(100,MAX_FILTERS*8*8)
        self.fcbn = nn.BatchNorm2d(MAX_FILTERS)
        # Conv layers for generator

        # list of channels for different filters
        ch = [256,128,64,3]

        # 8 -> 16 -> 32
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(len(ch)-1):
            self.conv.append(self.myconv(ch[i],ch[i+1]))
            self.bn.append(nn.BatchNorm2d(ch[i+1]))

        self.up = nn.Upsample(scale_factor=2)

    # Custom layer for upsample+conv
    def myconv(self,fi,fo,k=3,s=1,p=1):
        if deconv_mode == "upsample":
            return nn.Conv2d(fi,fo,k,s,p)
        elif deconv_mode == "transpose":
            return nn.ConvTranspose2d(fi,fo,k,2,p)
        else:
            print("Invalid deconv mode")
            return -1


    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1,MAX_FILTERS,8,8)
        x = self.fcbn(x)
        x = F.relu(x)

        for i, (conv, bn) in enumerate(zip(self.conv,self.bn)):
            #if deconv_mode == "upsample":
                #if i != len(self.conv) - 1: x = self.up(x)
            x = self.up(x)
            x = conv(x)
            if i != len(self.conv) - 1:
                x = bn(x)
                x = F.relu(x)

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
        ch = [3,64,128,256]

        # 64 -> 32 -> 16 -> 8
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(len(ch)-1):
            self.conv.append(self.myconv(ch[i],ch[i+1]))
            self.bn.append(nn.BatchNorm2d(ch[i+1]))

        self.validity = nn.Linear(MAX_FILTERS*8*8,1)

    # Custom conv layer for my purposes
    # Takes filters in, filters out
    def myconv(self,fi,fo,k=4,s=2,p=1):
        return nn.Conv2d(fi,fo,k,s,p)

    def forward(self,x):
        for i, (conv, bn) in enumerate(zip(self.conv,self.bn)):
            x = conv(x)
            
            if i != len(self.conv) - 1:
                x = bn(x)
                x = nn.LeakyReLU(0.2,inplace=True)(x)
                if use_dropout:
                    x = nn.Dropout2d(0.25)(x)
                

        x = x.view(-1,MAX_FILTERS*8*8)
        x = self.validity(x)
        return x


# Loss functions are defined below:

# First we have the Wasserstein loss which requires we label
# Real samples: -1, Fake: 1 for discriminator output
# Discriminator output should be linear as well (not sigmoidal)
# This will try to minimize distance between outputs and desired outputs
def W_Loss(y, y_real):
    return torch.mean(y * y_real)

# Gradient penalty (don't know the math behind this but it works)
# D is the discriminator model, avg_samp is _, gp_weight is the weight
# given to this loss function when training
def GP_Loss(D, y, y_real):
    alpha = torch.rand(batch_size, 1)
    # Notes on following expression:
    # We want A to be expanded and have the same size
    # As a batch of samples flattened
    # y_real.numel()/batch_size is the num of total elements in a single sample
    # contiguous is weird but basically just makes the .expand() operation
    # actually work????
    # afterwards we change alpha to have the actual same size
    # as the batch of samples (.view)
    alpha = alpha.expand(batch_size,
                         int(y_real.numel()/batch_size)).view(batch_size,
                                                              channels,
                                                              img_size,
                                                              img_size)
    if cuda: alpha = alpha.cuda()
    inp = alpha*y_real + (1-alpha)*y
    if cuda: inp = inp.cuda()
    inp = autograd.Variable(inp, requires_grad=True)
    D_inp = D(inp)
    outputs = torch.ones(D_inp.size()).float()
    if cuda: outputs = outputs.cuda()
    grads = autograd.grad(outputs=D_inp,
                          inputs=inp,
                          grad_outputs=outputs,
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    gp = ((grads.norm(2,dim=1) - 1) ** 2).mean() * gp_weight
    return gp

    
# MODELS INITIALIZED
G = Generator()
D = Discriminator()

if not cuda:
    summary(G, (100,),device="cpu")
    summary(D, (3,64,64),device="cpu")
else:
    G.cuda()
    D.cuda()
    summary(G, (100,))
    summary(D, (3, 64, 64))



G.apply(weights_init_normal)
D.apply(weights_init_normal)


d_opt = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.9))
g_opt = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.9))


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
    X = torch.from_numpy(X).float()
    if cuda:
        X = X.cuda()
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


def train(iterations):

    # make labels
    fake = torch.tensor(1.0)
    valid = -1*fake

    if cuda:
        fake = fake.cuda()
        valid = valid.cuda()
    
    G.train()
    D.train()
    for i in range(iterations):
        # Unfreeze discriminator
        for p in D.parameters():
            p.requires_grad = True
            
        # TRAIN DISCRIMINATOR
        for n in range(n_critic):
            # Sample from data
            np.random.shuffle(training_data)
            batch = torch.from_numpy(training_data[0:batch_size]).float()
            if cuda:
                batch = batch.cuda()
            
            # forward
            noise = torch.randn(batch_size,100)
            if cuda:
                noise = noise.cuda()
            gen_img = G(noise)
            gen_labels = D(gen_img) # Fake (output 1)
            batch_labels = D(batch) # Real (output -1)

            d_opt.zero_grad()
            # Train on real samples
            d_real = batch_labels.mean()
            d_real.backward(valid)

            # Train on fake samples
            d_fake = gen_labels.mean()
            d_fake.backward(fake)

            # Train with GP
            gp = GP_Loss(D, gen_img, batch)
            gp.backward()

            d_loss = d_fake - d_real + gp
            wass_dist = d_real - d_fake
            d_opt.step()

        # Freeze discriminator: 
        for p in D.parameters():
            p.requires_grad = False # prevents pointless calculations
            
        # TRAIN GENERATOR

        # forward
        noise = torch.randn(batch_size,100).float()
        if cuda:
            noise = noise.cuda()
        gen_img = G(noise)
        labels = D(gen_img).mean()
        
        # backward
        g_opt.zero_grad()
        labels.backward(valid)
        g_loss = -labels
        g_opt.step()
        
        print(i,"G loss:",g_loss.item(),"D loss:",d_loss.item())

        
        if (i+1)%sample_interval == 0:
            ShowSamples(str(i))
        if (i+1)%checkPoint_interval == 0:
            torch.save(G.state_dict(),"Gparams.pt")
            torch.save(D.state_dict(),"Dparams.pt")



train(iters)
