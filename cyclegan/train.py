import datasets
from models import CycleGAN
from constants import *
from ops import Tensor, npimage
import numpy as np
import torch
import matplotlib.pyplot as plt

A = datasets.GetTrainingData("celeba","tanh",(IMG_SIZE,IMG_SIZE),CHANNELS)
B = datasets.GetTrainingData("celeba","tanh",(IMG_SIZE,IMG_SIZE),CHANNELS)

A = np.moveaxis(A,3,1)
B = np.moveaxis(B,3,1)

A = torch.from_numpy(A)
B = torch.from_numpy(B)

data_sizes = [list(A.shape)[0],list(B.shape)[0]]

# Returns batch in form of (A,B)
def get_batch(batch_size):
	indsA = torch.randint(0,data_sizes[0],(batch_size,))
	indsB = torch.randint(0,data_sizes[1],(batch_size,))

	return Tensor(A[indsA]),Tensor(B[indsB])


model = CycleGAN()
if USE_CUDA: model.cuda()

if LOAD_CHECKPOINTS:
	model.load_checkpoint()

# Rendering stuff
fig,axs = plt.subplots(4,4)
# Draws samples
def save_samples(title):
	A_sample, B_sample = get_batch(4)
	B_fake, A_fake = model.A_to_B(A_sample), model.B_to_A(B_sample)

	print(torch.max(A_sample))
	print(torch.min(A_sample))
	print(torch.max(B_fake))
	print(torch.min(B_fake))
	A_sample = npimage(A_sample)
	B_sample = npimage(B_sample)
	A_fake = npimage(A_fake)
	B_fake = npimage(B_fake)

	for r in range(4):
		axs[r][0].imshow(A_sample[r])
		axs[r][1].imshow(B_fake[r])
		axs[r][2].imshow(B_sample[r])
		axs[r][3].imshow(A_fake[r])

	plt.savefig(title+".png")

# Actual training loop
for ITER in range(ITERATIONS):
	
	# Train discriminators
	for i in range(N_CRITIC):
		A_batch,B_batch = get_batch(BATCH_SIZE)
		D_A_loss,D_B_loss = model.train_disc_on_batch(A_batch,B_batch)
	
	# Train generators
	A_batch,B_batch = get_batch(BATCH_SIZE)
	G_A_loss, G_B_loss = model.train_gen_on_batch(A_batch, B_batch)
	
	# Write things down
	print("[",ITER,"/",ITERATIONS,"]")
	print("D A Loss:",D_A_loss)
	print("G A Loss:",G_A_loss)
	print("D B Loss:",D_B_loss)
	print("G B Loss:",G_B_loss)

	if ITER % SAMPLE_INTERVAL == 0:
		save_samples(str(ITER))
	if ITER % CHECKPOINT_INTERVAL == 0:
		model.save_checkpoint()
