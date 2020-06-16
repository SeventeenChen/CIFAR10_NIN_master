# -*- coding: utf-8 -*-

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from model import *
from data_loader import *
import os

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(comment='NIN')
checkpoint_dir = './results'
dummy_input = torch.ones([128, 3, 32, 32])

##########################
### SETTINGS
##########################

# Hyperparameters
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
NUM_EPOCHS = 150
MIN_TEST_ACC= 88

# Architecture
NUM_FEATURES = 32*32
NUM_CLASSES = 10

# Other
GRAYSCALE = False


def compute_acc(model, data_loader, device):
	correct_pred, num_examples = 0, 0
	model.eval()
	for i, (features, targets) in enumerate(data_loader):
		features = features.to(device)
		targets = targets.to(device)

		logits, probas = model(features)
		_, predicted_labels = torch.max(probas, 1)
		num_examples += targets.size(0)
		assert predicted_labels.size() == targets.size()
		correct_pred += (predicted_labels == targets).sum()
	return correct_pred.float() / num_examples * 100



if __name__ == '__main__':

	model = NIN(NUM_CLASSES)
	model.to(DEVICE)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
	iter_num = 0

	for epoch in range(NUM_EPOCHS):

		model.train()
		for batch_idx, (features, targets) in enumerate(train_loader):
			iter_num += 1
			features = features.to(DEVICE)
			targets = targets.to(DEVICE)

			### FORWARD AND BACK PROP
			logits, probas = model(features)
			cost = F.cross_entropy(logits, targets)
			writer.add_scalar('Train Loss', cost, iter_num)
			optimizer.zero_grad()

			cost.backward()

			### UPDATE MODEL PARAMETERS
			optimizer.step()


			if not batch_idx % 120:
				print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
					  f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
					  f' Cost: {cost:.4f} ')

		model.eval()
		with torch.set_grad_enabled(False):  # save memory during inference

			train_acc = compute_acc(model, train_loader, device=DEVICE)
			valid_acc = compute_acc(model, valid_loader, device=DEVICE)
			writer.add_scalar('Train Acc', train_acc, iter_num)
			writer.add_scalar('Valid Acc', valid_acc, iter_num)

			print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d}\n'
				  f'Train ACC: {train_acc:.2f} | Validation ACC: {valid_acc:.2f}')

			ckpt_model_filename = 'ckpt_valid_tacc_{}.pth'.format(valid_acc)
			ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)  # model_save

			if valid_acc > MIN_TEST_ACC:
				torch.save(model.state_dict(), ckpt_model_path)
				print("\nDone, save model at {}", ckpt_model_path)
				MIN_TEST_ACC = valid_acc


	with SummaryWriter(comment='NIN') as w:
		w.add_graph(model.to(DEVICE), (dummy_input.to(DEVICE),), True)



