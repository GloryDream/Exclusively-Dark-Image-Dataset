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
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--img_dir', type=str, help='The path of the result images')
parser.add_argument('--anno', type=str, help='The path of the cls_anno')
opt = parser.parse_args()
print(opt)


def file_name(file_dir, form='jpg'):
	file_list = []
	for root, dirs, files in os.walk(file_dir):
		for file in files:
			if os.path.splitext(file)[1] == '.' + form:
				file_list.append(os.path.join(root, file))
	return sorted(file_list)


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

	# dataset = EdarkDataset(root='/home/xinyu/dataset/Exclusively-Dark-Image-Dataset/ExDark',
	#                                      transform=transform,  mode='test')
	# dataloader = DataLoader(dataset,
	#                         batch_size=opt.batch_size,
	#                         shuffle=True,
	#                         num_workers=4)

	total = 0
	correct = 0
	for img_name in tqdm(img_name_list):
		img = Image.open(img_name)
		img = transform(img)
		img = img[None, :, :, :]
		#print(np.shape(img))
		output = resnet50(img.type(torch.cuda.FloatTensor))
		predicted = torch.argmax(output.data, 1).item()
		total += 1
		#print(predicted)
		if cls_anno[img_name.split('/')[-1]] == predicted:
			correct += 1
	print('Accuracy of the network on the %d test images: %d %%' % (total,
			100 * correct / total))



