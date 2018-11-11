import torchvision.models as models
from edark_image_net_dataset import EdarkDataset
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
from tqdm import tqdm
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--topk', type=int, help='The topk acc')
opt = parser.parse_args()
print(opt)

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
	dataset = EdarkDataset(root='/home/xinyu/dataset/Exclusively-Dark-Image-Dataset/ExDark',
	                                     transform=transform,  mode='test')
	dataloader = DataLoader(dataset,
	                        batch_size=opt.batch_size,
	                        shuffle=True,
	                        num_workers=4)

	total = 0
	correct = 0
	for img, label, _ in tqdm(dataloader):
		output = resnet50(img.type(torch.cuda.FloatTensor))
		total += 1

		_, top_idx = torch.topk(output.data, k=opt.topk, dim=1)
		top_idx = top_idx.cpu().numpy().tolist()[0]

		gd = ft_cls2idx[label[0]]
		if not isinstance(gd, list):
			gd = [gd]
		if list(set(gd)&set(top_idx)) == []:
			continue
		else:
			correct += 1
	print('Accuracy of the network on the %d test images: %f %%' % (total,
			100 * correct / total))



