import torch
from torch import nn
import torch.nn.functional as F
from layers import *
from math import floor, ceil 
from ops import weighted_sum

# Progressive generator
class ProgGen(nn.Module):
	def __init__(self):
		super(ProgGen, self).__init__()
	
		ch = [512,512,512,256,128,64]	
		self.current_progress = 0
		self.max_progress = len(ch)-2 
		self.blocks = nn.ModuleList()
		self.fc = nn.Linear(100,512*4*4)
	
		for i in range(len(ch)-1):
			self.blocks.append(GenConv(ch[i],ch[i+1],name="GenConv"+str(i)))
		
	def trans_info(self):
		# Current progress tells us last layer being used
		# If its .5 that means second last still transitioning
		return False if (self.current_progress % 1 == 0) else True
	
	def grow(self):
		# Do nothing if already grown up
		if self.current_progress == self.max_progress: return

		self.current_progress += 0.5 # Progress
			
		transitioning = self.trans_info()

		if transitioning:
			# Takes block to phase 1
			self.blocks[floor(self.current_progress)].grow()
		else:
			# Take block thats at the front and brings it to phase 1
			# Also adds new block 
			self.blocks[floor(self.current_progress)-1].grow()

	# Brings to max size
	def fullgrow(self):
		self.current_progress = self.max_progress

	def forward(self,x):
		# If transitioning, we will do a weighted sum if upsampled
		# Output of transitioning layer and output of new layer
		transitioning = self.trans_info()
		
		x = self.fc(x)
		x = x.view(-1,512,4,4)
		# If current progress is max progress, this loop does everything
		for i in range(floor(self.current_progress)+1):
			x = self.blocks[i](x)
		
		if transitioning:
			assert self.blocks[floor(self.current_progress)].RGBphase == 1
			# If we're transitioning, x should be a tuple now
			x,z = x	
			x = self.blocks[ceil(self.current_progress)](x)
			x = weighted_sum(z,x)

		return x		
		
# Progressive discriminator
class ProgDisc(nn.Module):
	def __init__(self):
		super(ProgDisc, self).__init__()
		
		ch = [64,128,256,512,512]
				
		self.current_progress = 0
		self.max_progress = len(ch)-2
		self.blocks = nn.ModuleList()

		for i in range(len(ch)- 1):
			self.blocks.append(DiscConv(ch[i],ch[i+1],name="DiscConv"+str(i)))

		self.fc = nn.Linear(512*4*4,1)

	# Tells us whether a layer as transitioning or not
	def trans_info(self):
		return False if self.current_progress % 1 == 0 else True

	def grow(self):
		if self.current_progress == self.max_progress: return

		self.current_progress += 0.5

		# This is what we have just entered (not what we WERE in)
		transitioning = self.trans_info()

		if transitioning:
			# Enter layer into transition phase
			ind = -1*floor(self.current_progress) - 1
			self.blocks[ind].grow()
		else:
			# Make last transitioning layer exit transition phase
			ind = -1*(floor(self.current_progress) - 1) - 1
			self.blocks[ind].grow()

	def fullgrow(self):
		self.current_progress = self.max_progress

	def forward(self,x):
		
		transitioning = self.trans_info()

		# This gets confusing cause its reverse of how it was for generator
		# Suppose we feed image into discriminator
		# If layer is transitioning, it will be the second layer
		if transitioning:
			# If second layer is transitioning
			# It should receive weighted sum of first layer output (x)
			# And downsampled input image (z)
			z = nn.MaxPool2d(2)(x)
			z = self.blocks[-1*(floor(self.current_progress))-1].rgb(z)
			x = self.blocks[-1*(ceil(self.current_progress))-1](x) 
			x = weighted_sum(z,x)

		# Start passing at -1*current_progress - 1'th layer
		for i in range(-1*floor(self.current_progress) -1,0):
			x = self.blocks[i](x)

		x = x.view(-1,512*4*4)
		x = self.fc(x)
	
		return x

# AND FINALLY, THE ACTUAL MODEL
class ProgGan(nn.Module):
	def __init__(self):
		super(ProgGan,self).__init__()
		
		self.progress = 0
		
		self.gen = ProgGen()
		self.disc = ProgDisc()
	
		self.max_progress = self.gen.max_progress

		self.optG = torch.optim.Adam(self.gen.parameters(),lr=LEARNING_RATE,betas=(0.5,0.9))
		self.optD = torch.optim.Adam(self.disc.parameters(),lr=LEARNING_RATE,betas=(0.5,0.9))

	def grow(self):
		if self.progress != self.max_progress: self.progress += 1
		self.gen.grow()
		self.disc.grow()
		
	def fullgrow(self):
		self.progress = self.max_progress
		self.gen.fullgrow()
		self.disc.fullgrow()

	# n is number of samples to generate
	def generate(self,n):
		x = torch.randn(n,100)
		return self.gen(x)

	def save_checkpoint(self):
		torch.save(self.state_dict(),"params.pt")

	def load_checkpoint(self):
		try:
			self.load_state_dict(torch.load("params.pt"))
			print("Loaded checkpoint")
		except:
			print("Could not load checkpoint")
