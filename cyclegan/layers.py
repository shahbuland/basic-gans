import torch
from torch import nn
from torch.nn import functional as F

# Decoding Block
class DecodeBlock(nn.Module):
	def __init__(self,fi,fo,k=3,s=1,p=1,useBN = True,Act=F.relu):
		super(DecodeBlock,self).__init__()
		
		self.conv = nn.Conv2d(fi,fo,k,s,p)
		self.bn = nn.BatchNorm2d(fo) if useBN else None
		self.Act = Act
		
	def forward(self,x):
		x = F.interpolate(x,scale_factor=(2,2))
		x = self.conv(x)
		x = self.Act(x)
		if self.bn is not None: x = self.bn(x)
		return x


# Encoding Block
class EncodeBlock(nn.Module):
	def __init__(self,fi,fo,k=3,s=1,p=1,useBN = True,Act = F.relu):
		super(EncodeBlock, self).__init__()
		
		self.conv = nn.Conv2d(fi,fo,k,s,p)
		self.pool = nn.MaxPool2d(2)
		self.bn = nn.BatchNorm2d(fo) if useBN else None
		self.Act = Act

	def forward(self,x):
		x = self.conv(x)
		x = self.pool(x)
		x = self.Act(x)
		if self.bn is not None: x = self.bn(x)
		return x

	
