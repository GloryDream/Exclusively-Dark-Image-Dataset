from torch.utils.data import Dataset
import pickle
import os
from PIL import Image


class EdarkDataset(Dataset):
	def __init__(self, root, transform=None, mode='train'):
		if mode=='train':
			with open(os.path.join(root, 'filtered_train_img_names.pkl'), 'rb') as f:
				self.img_names = pickle.load(f)
		elif mode == 'test':
			with open(os.path.join(root, 'filtered_test_img_names.pkl'), 'rb') as f:
				self.img_names = pickle.load(f)
		else:
			raise NotImplementedError
		self.transform = transform

		self.cls2imagenet_idx = {
			'Bicycle': 444,
			'Boat': 833,
			'Bottle': [440, 720, 737, 898, 907],
			'Bus': 779,
			'Car': [817, 864],
			'Cat': [281, 282, 283, 284, 285, 383],
			'Chair': [423, 559, 765],
			'Cup': [647, 968],
			'Dog': list(range(150, 276)),
			'Motorbike': 670,
			'People': 1000,
			'Table': [736, 532]
		}
		self.imagenet_idx2cls = {}
		for item in self.cls2imagenet_idx:
			if isinstance(self.cls2imagenet_idx[item], list):
				for id in self.cls2imagenet_idx[item]:
					self.imagenet_idx2cls[id] = item
			else:
				self.imagenet_idx2cls[self.cls2imagenet_idx[item]] = item

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self, idx):
		img_path = os.path.join('/home/xinyu/dataset/Exclusively-Dark-Image-Dataset/ExDark', self.img_names[idx])
		img = Image.open(img_path)
		cls = self.img_names[idx].split('/')[1]

		if self.transform:
			img = self.transform(img)
		return img, cls, img_path
