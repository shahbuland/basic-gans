import torch
from torch import autograd
from constants import *
import torch.nn.functional as F
from math import ceil

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def weighted_sum(x,y,ALPHA):
	return ((1-ALPHA)*x + (ALPHA*y))

def image_size_from_progress(prog):
	return 8*(2**ceil(prog))

# Copied from my previous implementation, still don't know what it does
def GP_Loss(D,y,y_real,prog):
	prog = ceil(prog)
	alpha = torch.rand(BATCH_SIZE,1)
	alpha = alpha.expand(BATCH_SIZE,int(y_real.numel()/BATCH_SIZE))
	size = image_size_from_progress(prog)
	alpha = alpha.view(-1,CHANNELS,size,size)
	if USE_CUDA: alpha = alpha.cuda()

	inp = alpha*y_real + (1-alpha)*y
	out = D(inp)
	valid = torch.ones(out.size()).float()
	if USE_CUDA: valid = valid.cuda()

	grads = autograd.grad(outputs=out,
						  inputs=inp,
						  grad_outputs=valid,
						  create_graph=True,retain_graph=True,only_inputs=True)[0]
	gp = ((grads.norm(2,dim=1) - 1) ** 2).mean() * GP_WEIGHT 
	return gp

# corrects batch by bringing down image size for the models progression
def correct_batch(batch, prog):
	prog = ceil(prog)
	desired_size = image_size_from_progress(prog)
	batch = F.interpolate(batch, size=(desired_size,desired_size))
	return batch
