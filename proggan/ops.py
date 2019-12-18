import torch
from constants import *

def weighted_sum(x,y):
	return ((1-ALPHA)*x + (ALPHA*y))
