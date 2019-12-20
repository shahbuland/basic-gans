from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from layers import EncodeBlock, DecodeBlock
from constants import *
from ops import get_labels, Tensor

# Generator 
class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		
		# Generator is an encoder-decoder
		# Assuming input is 64x64:
		ech = [CHANNELS,64,128,256,512]

		self.encode_blocks = nn.ModuleList()

		for i in range(len(ech)-1):
			self.encode_blocks.append(EncodeBlock(ech[i],ech[i+1],Act=nn.LeakyReLU(0.2)))

		# Now 512x4x4

		dch = [512,256,128,64,CHANNELS]

		self.decode_blocks = nn.ModuleList()

		for i in range(len(dch) - 2):
			self.decode_blocks.append(DecodeBlock(dch[i],dch[i+1]))
	
		self.decode_blocks.append(DecodeBlock(dch[-2],dch[-1],Act=torch.tanh,useBN=False))

		# Now CHANNELS x 64 x 64

	def forward(self, x):
		for block in self.encode_blocks:
			x = block(x)

		for block in self.decode_blocks:
			x = block(x)
		
		return x

# Discriminator
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		# Just encoder
		# Assume input is 64x64
		# Ends in 1 for 4x4 patch
		ch = [CHANNELS,64,128,256,1]

		self.convblocks = nn.ModuleList()

		for i in range(len(ch) - 2):
			self.convblocks.append(EncodeBlock(ch[i],ch[i+1],Act=nn.LeakyReLU(0.2)))
		self.convblocks.append(EncodeBlock(ch[-2],ch[-1],Act=torch.sigmoid))

	def forward(self,x):
		for block in self.convblocks:
			x = block(x)

		return x

# CycleGAN model
class CycleGAN(nn.Module):
	def __init__(self):
		super(CycleGAN,self).__init__()

		# Models
		self.models = nn.ModuleDict()
		self.models["G_BA"] = Generator()
		self.models["G_AB"] = Generator()
		self.models["D_A"] = Discriminator()
		self.models["D_B"] = Discriminator()

		# Optimizers
		self.opts = {}
		for name, model in self.models.items():
			self.opts[name] = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,betas=BETAS)
	
	# Functions that simplify using all the models
	def A_to_B(self,A):
		return self.models["G_AB"](A)

	def B_to_A(self,B):
		return self.models["G_BA"](B)

	def judge_A(self,A):
		return self.models["D_A"](A)

	def judge_B(self,B):
		return self.models["D_B"](B)

	def load_checkpoint(self):
		try:
			self.load_state_dict(torch.load("params.pt"))
			print("Loaded checkpoint")
		except:
			print("Could not load checkpoint")
	
	def save_checkpoint(self):
		torch.save(self.state_dict(), "params.pt")

	def train_disc_on_batch(self,batch_A,batch_B):
		
		# Zero out all optimizers
		for opt in self.opts.values():
			opt.zero_grad()	

		# Make labels
		real_labels = get_labels(1,BATCH_SIZE)
		fake_labels = get_labels(0,BATCH_SIZE)

		# Generate fake images 
		fake_B = self.A_to_B(batch_A)
		fake_A = self.B_to_A(batch_B)
		
		# Judge images
		real_score_A = self.judge_A(batch_A)
		fake_score_A = self.judge_A(fake_A)
		real_score_B = self.judge_B(batch_B)
		fake_score_B = self.judge_B(fake_B)

		loss = nn.MSELoss()

		# Calculate both losses, go backwards then step
		D_A_loss = (loss(real_score_A, real_labels) + loss(fake_score_A, fake_labels))
		D_B_loss = (loss(real_score_B, real_labels) + loss(fake_score_B, fake_labels))
		
		D_A_loss.backward()
		D_B_loss.backward()

		self.opts["D_A"].step()
		self.opts["D_B"].step()

		return D_A_loss.item(), D_B_loss.item()

	def train_gen_on_batch(self,batch_A,batch_B):
		
		# Zero out all optimizers
		for opt in self.opts.values():
			opt.zero_grad()

		# Make labels (we only want real in this case)
		real_labels = get_labels(1,BATCH_SIZE)

		# Generate fake images
		fake_B = self.A_to_B(batch_A)
		fake_A = self.B_to_A(batch_B)

		# Get scores
		score_A = self.judge_A(fake_A)
		score_B = self.judge_B(fake_B)

		loss = nn.MSELoss()
		
		# Get losses for single forward
		A_loss_fwd = loss(score_A, real_labels) # Loss for G_BA
		B_loss_fwd = loss(score_B, real_labels) # Loss for G_AB
		
		# Now cycle
		rec_A = self.B_to_A(fake_B)
		rec_B = self.A_to_B(fake_A)

		score_rec_A = self.judge_A(rec_A)
		score_rec_B = self.judge_B(rec_B)

		# Get cycle loss
		cycle_A_loss = CYCLE_WEIGHT*loss(score_rec_A, real_labels)
		cycle_B_loss = CYCLE_WEIGHT*loss(score_rec_B, real_labels)
	
		A_loss = A_loss_fwd + cycle_A_loss
		B_loss = B_loss_fwd + cycle_B_loss

		A_loss.backward() 
		B_loss.backward()
		
		self.opts["G_AB"].step()
		self.opts["G_BA"].step()

		return A_loss.item(), B_loss.item()
