from torch.utils.data import Dataset
import pickle
from imageio import imread
import os
import torch

cls2idx = {
	'Bicycle': 0,
	'Boat': 1,
	'Bottle': 2,
	'Bus': 3,
	'Car': 4,
	'Cat': 5,
	'Chair': 6,
	'Cup': 7,
	'Dog': 8,
	'Motorbike': 9,
	'People': 10,
	'Table': 11
}


class EdarkDataset(Dataset):
	def __init__(self, root, transform=None):
		with open(os.path.join(root, 'filtered_train_img_names.pkl'), 'rb') as f:
			self.img_names = pickle.load(f)
		self.transform = transform

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self, idx):
		img = imread(self.img_names[idx])
		cls_id = int(cls2idx[self.img_names[idx].split('/')[1]])
		label = torch.zeros(len(cls2idx))
		label[cls_id] = 1
		if self.transform:
			img = self.transform(img)
		return img, label
