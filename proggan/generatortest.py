import models
import layers
import torch

g = models.ProgGen()
x = torch.ones(1,100)

for i in range(10):
	print(g(x).shape)
	g.grow()
	print(i, "Success")
