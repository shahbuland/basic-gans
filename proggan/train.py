import torch
from torch import nn
from models import ProgGan
import matplotlib.pyplot as plt
import numpy as np
from ops import *

# Load dataset
#training_data = datasets.GetTrainingData("x",shape=(64,64),norm_style="tanh")
#training_data = np.moveaxis(training_data,3,1)
#training_data = torch.from_numpy(training_data)
training_data = torch.randn(1000,3,64,64)
data_size = list(training_data.shape)[0]
print("Data shape:", list(training_data.shape))

model = ProgGan()

fig,axs = plt.subplots(4,4)

plt.ion()

def ind_to_coord(n):
	r = n // 4
	c = n % 4
	return r,c

def draw_samples():
	imgs = model.generate(16)
	plt.cla()
	for i in range(16):
		r,c = ind_to_coord(i)
		img = imgs[i].detach().cpu().numpy()
		img = np.moveaxis(img,0,2)
		img = 0.5*(img + 1)
		axs[r][c].imshow(img)
	plt.pause(0.00001)
	plt.draw()

def get_batch(size):
	rand_ind = torch.randint(0,data_size,(size,))
	return training_data[rand_ind].cuda() if USE_CUDA else training_data[rand_ind]

# Init weights
model.apply(weights_init_normal)

if LOAD_CHECKPOINTS:
	model.load_checkpoint()

# Labels
fake = torch.tensor(1.0)
if USE_CUDA: fake = fake.cuda()
real = -1*fake
model.train()

# Actual training loop
for ITER in range(ITERATIONS+1):
	
	# Unfreeze discriminator
	for p in model.disc.parameters():
		p.requires_grad = True

	# Train discriminator
	for n in range(N_CRITIC):
		real_img = correct_batch(get_batch(BATCH_SIZE),model.progress)
		noise = torch.randn(BATCH_SIZE,100)
		if USE_CUDA: noise = noise.cuda()
		real_labels = model.disc(real_img)
		fake_img = model.gen(noise)
		fake_labels = model.disc(fake_img)

		model.optD.zero_grad()
	
		# Train on real samples
		d_real = real_labels.mean()
		d_real.backward(real,retain_graph=True)

		# Train on fake samples
		d_fake = fake_labels.mean()
		d_fake.backward(fake,retain_graph=True)

		# Train with GP
		gp = GP_Loss(model.disc, fake_img, real_img,model.progress)
		gp.backward(retain_graph=True)

		d_loss = d_fake - d_real + gp
		wass_dist = d_real - d_fake
		model.optD.step()

	# Freeze discriminator
	for p in model.disc.parameters():
		p.requires_grad = False

	# Train generator
	noise = torch.randn(BATCH_SIZE,100).float()
	if USE_CUDA:
		noise = noise.cuda()
	fake_img = model.gen(noise)
	labels = model.disc(fake_img).mean()

	# backward
	model.optG.zero_grad()
	labels.backward(real)
	g_loss = -labels
	model.optG.step()

	print("[",ITER,"/",ITERATIONS,"]", "G Loss:", g_loss.item(), "D Loss:", d_loss.item())
	
	if (ITER+1)%GROW_INTERVAL == 0:
		model.grow()		
	if (ITER+1)%SAMPLE_INTERVAL == 0:
		draw_samples()	
	if (ITER+1)%CHECKPOINT_INTERVAL == 0:
		model.save_checkpoint()
