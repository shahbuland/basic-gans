import torch
from torch import nn
import torch.nn.functional as F
from constants import *

# Layer for converting to RGB (end of generator)
def toRGB(nn.Module)
	def __init__(self,fi):
		super(toRGB, self).__init__()
		self.conv = nn.Conv2d(fi,CHANNELS,1,1,0)

	def forward(self,x):
		return F.tanh(self.conv(x))
		
# Layer for generic Generator conv block
def GenConv(nn.Module):
	def __init__(self,fi,fo,k=4,s=1,p=1,Act=F.relu,useBN=True):
		super(GenConv, self).__init__()
		
		# Constant layer stuff
		self.conv = nn.Conv2d(fi,fo,k,s,p)
		self.Act = Act
		self.useBN = useBN
		if useBN: self.bn = nn.BatchNorm2d(fo)

		# To RGB (would be phased out)
		self.rgb = toRGB(fo)
		self.useRGB = True # Initialize to true

	def grow(self):
		self.useRGB = False

	def forward(self, x):
		x = F.interpolate(x,scale_factor=2)
		x = self.conv(x)
		x = self.Act(x)
		if self.useBN: x = self.bn(x)
		if self.useRGB: x = self.rgb(x)
		return x

# Layer for converting from RGB (start of discriminator)
def fromRGB(nn.Module)
	def __init__(self,fo):
		super(fromRGB, self).__init__()
		self.conv = nn.Conv2d(CHANNELS,fo,1,1,0)

	def forward(self,x):
		return self.conv(x)

# Layer for generic Discriminator conv block
def DiscConv(nn.Module):
	def __init__(self,fi,fo,k=4,s=1,p=0,Act=F.relu,useBN=True,useDP=True)
		super(DiscConv, self).__init__()
		
		# From RGB
		self.rgb = fromRGB(fi)
		# Unlike gen, disc uses phases
		# If we are in phase 0, we use from RGB,
		# If we are in phase 1, we do a weighted sum
		# If we are in phase 2, we do not use RGB
		self.RGBphase = 0
		self.sum = WeightedSum() # Only needed for RGBphase = 1

		# Constant conv stuff
		self.conv = nn.Conv2d(fi,fo,k,s,p)
		self.Act = Act
		self.useBN = useBN
		self.useDP = useDP
		if useBN: self.bn = nn.BatchNorm(fo)

	def grow(self):
		# Unlike gen, disc can have this called twice
		self.RGBphase += 1
		
	# Optional param y is only needed when RGBphase is 1
	# At which point we'd need input from last layer and original image downsampled
	def forward(self,x, y = None):
		if self.RGBphase == 0:
			x = self.rgb(x)
		elif self.RGBphase == 1:
			# At this point, assume x is input from last layer
			# And y is the original image downsampled
			x = self.sum(x,y)  
		x = self.conv(x)
		x = self.Act(x)
		if self.useBN: x = self.bn(x)
		if self.useDP: x = F.dropout(x,0.25)
	
		return x

# Takes input of old image and new image
def WeightedSum(nn.Module):
	def __init__(self):
		super(WeightedSum, self).__init__()

	def forward(self, x, y):
		return ((1- ALPHA) * x) + (ALPHA * y)
