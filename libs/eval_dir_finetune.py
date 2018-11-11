from resnet_finetune import resnet50
import argparse
from torchvision import transforms
from tqdm import tqdm
import torch
import os
from PIL import Image
import json
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--img_dir', type=str, help='The path of the result images')
parser.add_argument('--anno', type=str, help='The path of the cls_anno')
parser.add_argument('--topk', type=int, help='The topk acc')

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
			'People': 1000,
			'Table': [736, 532]
		}

imagenet_idx2cls = {}
for item in cls2imagenet_idx:
	if isinstance(cls2imagenet_idx[item], list):
		for id in cls2imagenet_idx[item]:
			imagenet_idx2cls[id] = item
	else:
		imagenet_idx2cls[cls2imagenet_idx[item]] = item

ft_cls2idx = {'Bicycle': 0,
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
  'Table': 11}


if __name__ == '__main__':
	with open(opt.anno) as f:
		cls_anno = json.load(f)
	model = resnet50(finetuned=True).cuda()
	model.eval()
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
		output = model(img.type(torch.cuda.FloatTensor))
		total += 1
		#print(predicted)
		if opt.topk is None:
			predicted = torch.argmax(output.data, 1).item()
			if predicted not in imagenet_idx2cls:
				continue
			if cls_anno[img_name.split('/')[-1]] == imagenet_idx2cls[predicted]:
				correct += 1
		else:
			_, top_idx = torch.topk(output.data, k=opt.topk, dim=1)
			top_idx = top_idx.cpu().numpy().tolist()[0]

			gd = ft_cls2idx[cls_anno[img_name.split('/')[-1]]]
			if not isinstance(gd, list):
				gd = [gd]
			if list(set(gd)&set(top_idx)) == []:
				continue
			else:
				correct += 1

	print('Accuracy of the network on the %d test images: %f %%' % (total,
			100 * correct / total))



