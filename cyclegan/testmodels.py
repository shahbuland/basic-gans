import models
import torch
from constants import *

g = models.Generator()
d = models.Discriminator()

x = torch.ones(1,CHANNELS,64,64)

y = g(x)
z = d(x)

print("G output shape:", list(y.shape))
print("D output shape:", list(z.shape))

