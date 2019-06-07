from __future__ import print_function
import argparse
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import PIL

from dcgan import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--netG', required=True, help="path to netG (to generate data)")
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake | gray_folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--algo', type=str, default='drunk', help='drunk | linear | lerp')
parser.add_argument('--variation', type=float, default=0.1, help='variation level (for drunk and linear algorithms)')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of images to generate')
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def gray_loader(path):
    return PIL.ImageOps.grayscale(PIL.Image.open(path))

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset in ['gray_folder']:
    # folder dataset (grayscale)
    dataset = dset.ImageFolder(root=opt.dataroot,
                           loader=gray_loader,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.CenterCrop(opt.imageSize),
#                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
    nc=1
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
imageSize = int(opt.imageSize)

netG = Generator(ngpu, ngf, nc, imageSize, nz).to(device)
# netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Generator n. params: {}".format(count_parameters(netG)))


noise_start = torch.randn(opt.batchSize, nz, 1, 1, device=device)

noise_end = torch.randn(opt.batchSize, nz, 1, 1, device=device)
noise_add = torch.randn(opt.batchSize, nz, 1, 1, device=device) * opt.variation

z = noise_start.clone().detach()

import progressbar

bar = progressbar.ProgressBar(maxval=opt.niter, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in range(opt.niter):
    bar.update(i+1)
    fake = netG(z)
    images = fake.detach()
    # Generate big image file.
    vutils.save_image(images,
                      '%s/images_%03d.png' % (opt.outf, i),
                      normalize=True)
    # Generate all images in sub folders.
    for b in range(opt.batchSize):
        image_dir = '%s/images_%02d' % (opt.outf, b)
        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)
        vutils.save_image(images[b],
                              '%s/image_%02d_%03d.png' % (image_dir, b, i),
                              normalize=True)

    # Add some noise.
    if opt.algo == 'drunk':
        z += torch.randn(opt.batchSize, nz, 1, 1, device=device) * opt.variation
    elif opt.algo == 'linear':
        z += noise_add.clone().detach()
    elif opt.algo == 'lerp':
        t = (i+1) / float(opt.niter)
        z = (1-t) * noise_start + t * noise_end
    else:
        print('Unrecognized algorithm: {}' % opt.algo)
        exit()
bar.finish()