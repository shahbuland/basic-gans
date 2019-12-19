import torch
from torch import nn
import torch.nn.functional as F
from constants import *
from ops import weighted_sum

# Layer for converting to RGB (end of generator)
class toRGB(nn.Module):
	def __init__(self,fi):
		super(toRGB, self).__init__()
		self.conv = nn.Conv2d(fi,CHANNELS,1,1,0)

	def forward(self,x):
		return torch.tanh(self.conv(x))
		
# Layer for generic Generator conv block
class GenConv(nn.Module):
	def __init__(self,fi,fo,k=3,s=1,p=1,useBN=True, name=None):
		super(GenConv, self).__init__()
		
		# To track layers (useful for debugging)
		self.name = name
		
		# Constant layer stuff
		self.conv = nn.Conv2d(fi,fo,k,s,p)
		self.useBN = useBN
		if useBN: self.bn = nn.BatchNorm2d(fo)

		# To RGB (would be phased out)
		self.rgb = toRGB(fo)
		# if 0, this is last layer
		# if 1, this is learning to not be last layer
		# if 2, this is not last layer
		self.RGBphase = 0

	def grow(self):
		self.RGBphase += 1

	def forward(self, x):
		x = F.interpolate(x,scale_factor=2)

		x = self.conv(x)
		x = F.relu(x)
		if self.useBN: x = self.bn(x)
		if self.RGBphase == 0: # normal
			x = self.rgb(x)
		# In the below case, x becomes what we send next layer
		# z is what is passed to the previous layer
		elif self.RGBphase == 1: # Transitioning
			z = F.interpolate(x,scale_factor=2)
			z = self.rgb(z)
			return x,z # Next layer needs to get x,z
		# When 2, we do nothing

		return x

# Layer for converting from RGB (start of discriminator)
class fromRGB(nn.Module):
	def __init__(self,fo):
		super(fromRGB, self).__init__()
		self.conv = nn.Conv2d(CHANNELS,fo,1,1,0)

	def forward(self,x):
		return self.conv(x)

# Layer for generic Discriminator conv block
class DiscConv(nn.Module):
	def __init__(self,fi,fo,k=3,s=1,p=1,useBN=True,useDP=False,name=None):
		super(DiscConv, self).__init__()
		
		self.name = name

		# From RGB
		self.rgb = fromRGB(fi)
		# Unlike gen, disc uses phases
		# If we are in phase 0, we use from RGB,
		# If we are in phase 1, we do a weighted sum
		# If we are in phase 2, we do not use RGB
		self.RGBphase = 0

		# Constant conv stuff
		self.conv = nn.Conv2d(fi,fo,k,s,p)
		self.pool = nn.MaxPool2d(2)
		self.useBN = useBN
		self.useDP = useDP
		if useBN: self.bn = nn.BatchNorm2d(fo)

	def grow(self):
		# Unlike gen, disc can have this called twice
		self.RGBphase += 1
		
	def forward(self,x):
		if self.RGBphase == 0:
			x = self.rgb(x)
		# In any other RGBphase, work is done by model (not layer)
		x = self.conv(x)
		x = self.pool(x)
		x = F.leaky_relu(x,0.2)
		if self.useBN: x = self.bn(x)
		if self.useDP: x = F.dropout(x,0.25)
	
		return x
