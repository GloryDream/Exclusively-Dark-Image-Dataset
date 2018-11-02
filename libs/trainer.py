from resnet_finetune import resnet50
from edark_dataset import EdarkDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms, utils
import os
from tensorboardX import SummaryWriter
import argparse
import datetime
import dateutil.tz
from torchvision.utils import make_grid
from tqdm import tqdm

os.makedirs('output', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
# parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
# parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
# parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
# parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
# parser.add_argument('--latent_dim', type=int, default=48, help='dimensionality of the latent space')
opt = parser.parse_args()
print(opt)

# prepare dir
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
model_name = opt.name
output_dir = os.path.join('output', model_name)
logdir_path = os.path.join(output_dir, 'log', timestamp)
model_path = os.path.join(output_dir, 'model')

# mkdir
if not opt.reload:
	os.makedirs(output_dir)
	os.makedirs(logdir_path)
	os.makedirs(model_path)
TB = SummaryWriter(log_dir=logdir_path)


def save_checkpoint(state, filename):
	"""
	from pytorch/examples
	"""
	basename = os.path.dirname(filename)
	if not os.path.exists(basename):
		os.makedirs(basename)
	torch.save(state, filename)


def save(save_path, net, optimizer, epoch, ckpt_name):
	resume_file = os.path.join(save_path, ckpt_name)
	print('==>save', resume_file)
	save_checkpoint({
		'state_dict': net.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': epoch,
	}, filename=resume_file)


def save_model(base, fc, optimizer_base, optimizer_fc, epoch_flag, current_epoch):
	if epoch_flag == -1:
		g_ckpt_name = 'base_checkpoint.pth.tar'
		d_ckpt_name = 'fc_checkpoint.pth.tar'
	else:
		g_ckpt_name = 'base_checkpoint_' + str(current_epoch) + '.pth.tar'
		d_ckpt_name = 'fc_checkpoint_' + str(current_epoch) + '.pth.tar'
	save_path = model_path
	save(save_path, base, optimizer_base, current_epoch, g_ckpt_name)
	save(save_path, fc, optimizer_fc, current_epoch, d_ckpt_name)


def reload(base, fc, optimizer_base, optimizer_fc, epoch):
	restore_path = model_path

	if epoch == -1:
		base_ckpt_name = 'base_checkpoint.pth.tar'
		fc_ckpt_name = 'fc_checkpoint.pth.tar'
	else:
		base_ckpt_name = 'base_checkpoint_' + str(epoch) + '.pth.tar'
		fc_ckpt_name = 'fc_checkpoint_' + str(epoch) + '.pth.tar'

	base_resume_file = os.path.join(restore_path, base_ckpt_name)
	fc_resume_file = os.path.join(restore_path, fc_ckpt_name)

	if os.path.isfile(base_resume_file) and os.path.isfile(fc_resume_file):
		print("=> loading checkpoint '{}'".format(base_ckpt_name))
		base_checkpoint = torch.load(base_resume_file)
		base.load_state_dict(base_checkpoint['state_dict'])
		optimizer_base.load_state_dict(base_checkpoint['optimizer'])

		print("=> loading checkpoint '{}'".format(fc_ckpt_name))
		fc_checkpoint = torch.load(fc_resume_file)
		fc.load_state_dict(fc_checkpoint['state_dict'])
		optimizer_fc.load_state_dict(fc_checkpoint['optimizer'])

		epoch = base_checkpoint['epoch']
		print("=> loaded checkpoint (epoch {})".format(epoch))
	else:
		print("=> no checkpoint found at '{}'".format(base_ckpt_name))
		exit(0)

	return epoch


class trainer(object):
	def __init__(self):
		self.model = resnet50(pretrained=True, num_classes=12)
		self.optimizer_base = torch.optim.Adam(self.model.base.parameters(), lr=0.0001)
		self.optimizer_fc = torch.optim.Adam(self.model.base.parameters(), lr=0.001)

	def build_dataloader(self):
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
		transform = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])

		self.dataloader = DataLoader(EdarkDataset(root='~/dataset/Exclusively-Dark-Image-Dataset/ExDark',
												  transform=transform),
									 batch_size=opt.batch_size,
									 shuffle=True,
									 num_workers=4)

	def train(self):
		for epoch in tqdm(range(opt.n_epochs)):
			print("[Epoch %d/%d]" % (epoch, opt.n_epochs))


