import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt


base_path = "./datasets/"
# Given dataset name, returns data as a numpy array, assuming
# the path base_path/[dataset name] exists and is occupied with images
# norm_style states how to normalize the data
def GetTrainingData(dataset_name, norm_style=None,
                    shape=(64,64), channels=3):
    full_path = base_path+dataset_name+"/"
    if(not os.path.isdir(full_path)):
        print("No such directory found!")
        return
    X = []
    for path in os.listdir(full_path):
        data = Image.open(full_path+path)
        if(channels==1):
            data = data.convert("L")
        if(channels==3 and data.mode == "L"):
            data = data.convert("RGB")
        data = data.resize(shape)
        data = np.asarray(data)
        if(channels==1):
            data = np.expand_dims(data,2)
        if(norm_style=="sigmoid"):
            data = data/255
        if(norm_style=="tanh"):
            data = data/127.5 - 1
        X.append(data)
    return np.asarray(X)

  
# Convert data for numpy to data for torch (moves axes as required by torch models)
def torch(data):
    return torch.from_numpy(np.moveaxis(data,3,1))

# Samples r*c images from a dataset
def sample(dataset,r,c):
    n = r*c
    n_max = dataset.shape[0]
    inds = np.random.randint(0,n_max,size=n)
    imgs = dataset[inds]
    cnt = 0
    fig,axs = plt.subplots(r,c)
    for i in range(r):
        for j in range(c):
            axs[i][j].imshow(imgs[cnt])
            cnt+=1

    plt.savefig("real.png")
    plt.close()
    
    
