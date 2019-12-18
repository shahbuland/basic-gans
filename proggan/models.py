import torch
from torch import nn
import torch.nn.functional as F

class GenConvBlock(nn.Module):
	def __init__(self, C_in, C_out, K=4, S=1, P=1, BN=True, Up=True, Scale=2)
		super(GenConvBlock,self).__init__()
		self.BN = BN
		self.Up = Up
		if UP:
			self.up = lambda x : F.interpolate(x, scale_factor=(Scale,Scale))
		self.conv = nn.Conv2d(C_in,C_out,K,S,P)
		if BN:
			self.bn = nn.BatchNorm2d(C_out)
	
	# Act is specified in forward as it may change
	def forward(self, x, Act):
		if self.UP: x = self.up(x)
		x = self.conv(x)
		x = self.Act(x)
		if self.BN: x = self.bn(x)
		
		return x
		
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.conv_layers = nn.ModuleList()

		self.conv_layers.append(GenConvBlock(1,512,K=4,P=0,Up=False))
		# 512,4,4
		self.conv_layers.append(GenConvBlock(512,256))
		# 256,8,8
		self.conv_layers.append(GenConvBlock(256,128))
		# 128,16,16
		self.conv_layers.append(GenConvBlock(128,64))
		# 64,32,32
		self.conv_layers.append(GenConvBlock(64,CHANNELS,Act=F.torch,BN=False)
		# 3,64,64

		self.progress = INITIAL_PROGRESS # How many layers to use at start
		# Max progress in this case is 5

	def forward(self, x):
	
	for l in range(self.progress):
			
