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
		predicted = torch.argmax(output.data, 1).item()
		print(output.data.size())
		total += 1
		#print('label: ', label)
		#print('predicted: ', predicted)
		if opt.topk is None:
			if predicted not in dataset.imagenet_idx2cls:
				continue
			if dataset.imagenet_idx2cls[predicted] == label[0]:
				correct += 1
		else:
			_, tok_idx = torch.topk(predicted, k=opt.topk, dim=1)
			tok_idx = tok_idx.numpy().tolist()[0]

			gd = dataset.cls2imagenet_idx[label[0]]
			if not isinstance(gd, list):
				gd = [gd]
			if list(set(gd)&set(tok_idx)) == []:
				continue
			else:
				correct += 1
	print('Accuracy of the network on the %d test images: %f %%' % (total,
			100 * correct / total))



