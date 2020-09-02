import torch
import torch.nn as nn
from importlib import import_module
from utils.funcs import *

def mean_normalization(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size))/std.expand(size)
    return normalized_feat
    
#### Attention is all you need?
class Attention(nn.Module):
    def __init__(self, num_features):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.key_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.value_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.softmax = nn.Softmax(dim = -1)
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.uniform_(self.query_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.uniform_(self.key_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.value_conv.weight)
        nn.init.uniform_(self.value_conv.bias, 0.0, 1.0)
        
    def forward(self, content_feat, style_feat):
        Query = self.query_conv(mean_normalization(content_feat))
        Key = self.key_conv(mean_normalization(style_feat))
        Value = self.value_conv(style_feat)
        batch_size, channels, height, width = Query.size()
        Query = Query.view(batch_size, -1, width * height).permute(0, 2, 1)
        batch_size, channels, height, width = Key.size()
        Key = Key.view(batch_size, -1, width * height)
        Attention_Weights = self.softmax(torch.bmm(Query, Key))

        batch_size, channels, height, width = Value.size()
        Value = Value.view(batch_size, -1, width * height)
        Output = torch.bmm(Value, Attention_Weights.permute(0, 2, 1))
        batch_size, channels, height, width = content_feat.size()
        Output = Output.view(batch_size, channels, height, width)
        return Output

############################################## Custom Block #2.1 ##############################################
class MetaAdaIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.shared_weight = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        self.shared_bias = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        self.shared_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.gamma_conv = nn.Conv2d(num_features, num_features, (3, 3))
        self.beta_conv = nn.Conv2d(num_features, num_features, (3, 3))
        self.attention = Attention(num_features)
        self.relu = nn.ReLU()
        nn.init.ones_(self.shared_weight)
        nn.init.zeros_(self.shared_bias)
        nn.init.xavier_uniform_(self.gamma_conv.weight)
        nn.init.uniform_(self.gamma_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.beta_conv.weight)
        nn.init.uniform_(self.beta_conv.bias, 0.0, 1.0)

    def forward(self, content_feat, style_feat, output_shared=False):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_feat = self.attention(content_feat, style_feat)
        style_gamma = self.relu(self.gamma_conv(self.shared_pad(style_feat)))
        style_beta = self.relu(self.beta_conv(self.shared_pad(style_feat)))
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        shared_affine_feat = normalized_feat * self.shared_weight.view(1, self.num_features, 1, 1).expand(size) + \
                             self.shared_bias.view(1, self.num_features, 1, 1).expand(size)
        if output_shared:
            return shared_affine_feat
        output = shared_affine_feat * style_gamma + style_beta
        return output
        

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.MetaAdaIN4_1 = MetaAdaIN(512)
        self.MetaAdaIN5_1 = MetaAdaIN(512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        return self.merge_conv(self.merge_conv_pad(self.MetaAdaIN4_1(content4_1, style4_1) + self.upsample5_1(self.MetaAdaIN5_1(content5_1, style5_1))))


class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        #transform
        self.transform = Transform(in_planes = 512)
        self.decoder = decoder
        #Picking up training from in-between
        if(start_iter > 0):
            print("Loading pretrained models")
            self.transform.load_state_dict(torch.load('transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))
        
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def forward(self, content, style):
        ### GENERATE STYLIZED IMAGE ###
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        g_t = self.decoder(stylized)
        
        ### CONTENT AND STYLE LOSSES ###
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm = True) + self.calc_content_loss(g_t_feats[4], content_feats[4], norm = True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        
        ### IDENTITY LOSSES ###
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        #Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_identity1 = self.calc_content_loss(Icc, content) #+ self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        #Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) #+ self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) #+ self.calc_content_loss(Fss[i], style_feats[i])
        
        return loss_c, loss_s, l_identity1, l_identity2


import numpy as np
from torch.utils import data
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='./train_content/',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='./train_style/',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=2000)
parser.add_argument('--start_iter', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda')
writer = SummaryWriter(log_dir='experiments')

enc_dec = import_module("enc_dec")
decoder = enc_dec.decoder
vgg = enc_dec.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = Net(vgg, decoder, args.start_iter)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),num_workers=args.n_threads))
style_iter = iter(data.DataLoader(style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),num_workers=args.n_threads))

optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()}], lr=args.lr)

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))


print("content_wt: {}, style_wt: {}".format(args.content_weight, args.style_weight))
for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)
    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

writer.close()