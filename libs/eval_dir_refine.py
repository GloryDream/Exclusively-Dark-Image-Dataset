import torchvision.models as models
from edark_image_net_dataset import EdarkDataset
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
from tqdm import tqdm
import torch
import os
from PIL import Image
import json
import numpy as np
import collections


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--img_dir', type=str, help='The path of the result images')
parser.add_argument('--anno', type=str, help='The path of the cls_anno')
parser.add_argument('--topk', type=int, required=True, help='The topk acc')

opt = parser.parse_args()
print(opt)


def file_name(file_dir, form='jpg'):
	file_list = []
	for root, dirs, files in os.walk(file_dir):
		for file in files:
			if os.path.splitext(file)[1] == '.' + form:
				file_list.append(os.path.join(root, file))
	return sorted(file_list)

cls2imagenet_idx = {
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
			# 'People': 1000,
			'Table': [736, 532]
		}

imagenet_idx2cls = {}
for item in cls2imagenet_idx:
	if isinstance(cls2imagenet_idx[item], list):
		for id in cls2imagenet_idx[item]:
			imagenet_idx2cls[id] = item
	else:
		imagenet_idx2cls[cls2imagenet_idx[item]] = item

indices = list(imagenet_idx2cls.keys())
oreder_imagenet_idx2cls = collections.OrderedDict(sorted(imagenet_idx2cls.items()))

refined_imagenet_idx2cls = {}
for idx, (_, v) in enumerate(oreder_imagenet_idx2cls.items()):
	refined_imagenet_idx2cls[idx] = v

refined_imagenet_cls2idx = {}
for k, v in refined_imagenet_idx2cls.items():
	if v not in refined_imagenet_cls2idx:
		refined_imagenet_cls2idx[v] = [k]
	else:
		refined_imagenet_cls2idx[v].append(k)


if __name__ == '__main__':
	with open(opt.anno) as f:
		cls_anno = json.load(f)
	resnet50 = models.resnet50(pretrained=True).cuda()
	resnet50.eval()
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                 std=[0.229, 0.224, 0.225])
	transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
	img_name_list = file_name(opt.img_dir)

	total = 0
	correct = 0
	for img_name in tqdm(img_name_list):
		img = Image.open(img_name)
		img = transform(img)
		img = img[None, :, :, :]
		#print(np.shape(img))
		output = resnet50(img.type(torch.cuda.FloatTensor))
		output = output.data.cpu().numpy()[0]
		refined_output = np.take(output, indices)
		total += 1
		#print(predicted)

		top_idx = refined_output.argsort()[-opt.topk][::-1]

		if img_name.split('/')[-1] == 'People':
			continue
		gd = refined_imagenet_cls2idx[cls_anno[img_name.split('/')[-1]]]
		if not isinstance(gd, list):
			gd = [gd]
		if list(set(gd)&set(top_idx)) == []:
			continue
		else:
			correct += 1

	print('Top %d accuracy of the network on the %d test images: %f %%' % (opt.topk, total,
			100 * correct / total))



