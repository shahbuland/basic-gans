from models import ProgDisc,ProgGen
import torch

d = ProgDisc()
g = ProgGen()
x = torch.ones(1,100)

for i in range(7):
	y = g(x)
	print("Gen Output:",y.shape)
	print("Disc output:", d(y).shape)
	g.grow()
	d.grow()
