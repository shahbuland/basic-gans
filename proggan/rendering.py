import matplotlib.pyplot as plt
import numpy as np
import torch
from constants import *

# T is tensor, axs is axs to draw on
render(axs, T):
	assert len(list(T.shape)) == 3
	
	# Clear axis and change to number range for matplotlib
	axs.cla()
	if NORM = "tanh":
		T = 0.5*(T+1)
	
	# Grayscale
	if CHANNELS == 1:
		A = T.detach().cpu().squeeze().numpy()
		axs.imshow(A,cmap='gray')
	
	# RGB
	if CHANNELS == 3:
		A = T.detach().cpu().numpy()
		axs.imshow(A)


colours = ['red','green','blue','orange'] # List of colours for plots

# Graph object for tracking data
# We always assume first number in data is x axis, everything after is y
class Graph:
	# param_number is how many things graph should track
	# max_len is too keep graph from getting too long
	def __init__(self, param_number, max_len):
		self.param_number = param_number
		self.values = [[] for _ in range(param_number)]	
		self.max_len = max_len

	# Adds data to graph
	# Assumes data is param_number length list
	def add_data(self,data):
		for i in range(self.param_number):
			self.values[i].append(data[i])
		if(len(self.values[0]) > 100):
			for i in range(self.param_number):
				del self.values[i][0]

	def draw_graph(self,axs):
		data = [np.asarray(self.values[i]) for i in range(self.param_number)]
		axs.cla()
		horizontal = data[0]
		for i in range(1,self.param_number):
			axs.plot(horizontal, data[i], color=colours[i-1])

	
	
