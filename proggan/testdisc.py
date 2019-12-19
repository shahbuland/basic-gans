from models import ProgDisc,ProgGen
import torch
from math import floor,ceil

d = ProgDisc()
g = ProgGen()
x = torch.ones(1,100)

for i in range(7):
	y = g(x)
	print("Gen Output:",y.shape)
	print(8*(2**ceil(g.current_progress)))
	print("Disc output:", d(y).shape)
	g.grow()
	d.grow()
