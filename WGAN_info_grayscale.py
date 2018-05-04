import utils, torch, time, os, pickle, itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import datetime

from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from src.snlayers.snconv2d import SNConv2d
from src.snlayers.snlinear import SNLinear

len_continuous_code = 2
len_discrete_code = 2

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(generator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 62
            self.output_dim = 1
        elif dataset == 'celebA' or dataset == 'sdd':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 62
            self.output_dim = 1
        
        self.input_dim += (len_continuous_code + len_discrete_code)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input, cont_code, dist_code):
        x = torch.cat([input, cont_code, dist_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist'):
        super(discriminator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
        elif dataset == 'celebA' or dataset == 'sdd':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 1
            self.output_dim = 1

        self.conv = nn.Sequential(
            # nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            SNConv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(64, 128, 4, 2, 1),
            SNConv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            #nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            SNLinear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            #nn.Linear(1024, self.output_dim),
            SNLinear(1024, self.output_dim + len_continuous_code + len_discrete_code),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)
        
        # a = F.sigmoid(x[:, self.output_dim])
        a = x[:, self.output_dim - 1]
        b = x[:, self.output_dim:self.output_dim + len_continuous_code]
        c = x[:, self.output_dim + len_continuous_code:]

        return a, b ,c

class WGAN_info_grayscale(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.c = 0.01                   # clipping value
        self.n_critic = 5               # the number of iterations of the critic per generator iteration
        
        self.datetime = datetime.datetime.now().strftime("%Y-%B-%d-%I-%M%p")
        self.writer = SummaryWriter(os.path.join("./runs", self.dataset, self.model_name, self.datetime))

        # networks init
        self.G = generator(self.dataset)
        self.D = discriminator(self.dataset)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.info_optimizer = optim.Adam(itertools.chain(self.G.parameters(), self.D.parameters()), lr=args.lrD,
                                         betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # load dataset
        if self.dataset == 'mnist':
            self.data_loader = DataLoader(datasets.MNIST('data/mnist', train=True, download=True,
                                                         transform=transforms.Compose(
                                                             [transforms.ToTensor()])),
                                          batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'fashion-mnist':
            self.data_loader = DataLoader(
                datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()])),
                batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'celebA':
            self.data_loader = utils.load_celebA('data/celebA', transform=transforms.Compose(
                [transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]), batch_size=self.batch_size,
                                                 shuffle=True)
        elif self.dataset == 'sdd':
            self.data_loader = utils.load_sdd('./data/vgg_pets/processed/bounding_box_imgs', transform=transforms.Compose(
            [transforms.Grayscale(), transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor(), ]),
                                          batch_size=self.batch_size,
                                          shuffle=True)
        self.z_dim = 62
        self.y_dim = len_discrete_code
        
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(len_discrete_code):
            self.sample_z_[i * self.y_dim] = torch.rand(1, self.z_dim)
            for j in range(1, self.y_dim):
                self.sample_z_[i * self.y_dim + j] = self.sample_z_[i * self.y_dim]
                
        temp = torch.zeros((len_discrete_code, 1))
        for i in range(self.y_dim):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(len_discrete_code):
            temp_y[i * self.y_dim: (i + 1) * self.y_dim] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.y_dim))
        self.sample_y_.scatter_(1, temp_y.type(torch.LongTensor), 1)  # diagonal
        self.sample_c_ = torch.zeros((self.sample_num, len_continuous_code))
        
        # manipulating two continuous code
        temp_z_ = torch.rand((1, self.z_dim))
        self.sample_z2_ = temp_z_

        for i in range(self.sample_num - 1):
            self.sample_z2_ = torch.cat([self.sample_z2_, temp_z_])

        y = np.zeros(self.sample_num, dtype=np.int64)
        y_one_hot = np.zeros((self.sample_num, len_discrete_code))
        y_one_hot[np.arange(self.sample_num), y] = 1
        self.sample_y2_ = torch.from_numpy(y_one_hot).type(torch.FloatTensor)

        temp_c = torch.linspace(-1, 1, 10)
        self.sample_c2_ = torch.zeros((self.sample_num, len_continuous_code))
        
        # import IPython
        # IPython.embed()
        
        for i in range(10):
            for j in range(10):
                self.sample_c2_[i * 10 + j, 0] = temp_c[i]
                self.sample_c2_[i * 10 + j, 1] = temp_c[j]

        # fixed noise
        if self.gpu_mode:
            self.sample_z_, self.sample_y_, self.sample_c_, self.sample_z2_, self.sample_y2_, self.sample_c2_ = \
                Variable(self.sample_z_.cuda(), volatile=True), Variable(self.sample_y_.cuda(), volatile=True), \
                Variable(self.sample_c_.cuda(), volatile=True), Variable(self.sample_z2_.cuda(), volatile=True), \
                Variable(self.sample_y2_.cuda(), volatile=True), Variable(self.sample_c2_.cuda(), volatile=True)
        else:
            self.sample_z_, self.sample_y_, self.sample_c_, self.sample_z2_, self.sample_y2_, self.sample_c2_ = \
                Variable(self.sample_z_, volatile=True), Variable(self.sample_y_, volatile=True), \
                Variable(self.sample_c_, volatile=True), Variable(self.sample_z2_, volatile=True), \
                Variable(self.sample_y2_, volatile=True), Variable(self.sample_c2_, volatile=True)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['info_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        self.load()
            
        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                
                # NOTE: not supervised
                y_disc_ = torch.from_numpy(
                        np.random.multinomial(1, len_discrete_code * [float(1.0 / len_discrete_code)],
                                              size=[self.batch_size])).type(torch.FloatTensor)

                y_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.batch_size, 2))).type(torch.FloatTensor)
                
                if self.gpu_mode:
                    x_, z_, y_disc_, y_cont_ = Variable(x_.cuda()), Variable(z_.cuda()), \
                                               Variable(y_disc_.cuda()), Variable(y_cont_.cuda())
                else:
                    x_, z_, y_disc_, y_cont_ = Variable(x_), Variable(z_), Variable(y_disc_), Variable(y_cont_)
                    
                step = epoch * self.data_loader.dataset.__len__() // self.batch_size + iter
                self.writer.add_image('Real_image', vutils.make_grid(x_.data), step)

                # update D network
                self.D_optimizer.zero_grad()

                D_real, _, _ = self.D(x_)
                D_real_loss = -torch.mean(D_real)

                G_ = self.G(z_, y_cont_, y_disc_)
                D_fake, _, _ = self.D(G_)
                D_fake_loss = torch.mean(D_fake)

                D_loss = D_real_loss + D_fake_loss

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()
                
                self.writer.add_image('Fake_image', vutils.make_grid(G_.data), step)
                
                # fix noise
                samples = self.G(self.sample_z_, self.sample_c_, self.sample_y_)
                self.writer.add_image('Fake_fixed_image', vutils.make_grid(samples.data), step)

                # clipping D
                for p in self.D.parameters():
                    p.data.clamp_(-self.c, self.c)
                
                if ((iter+1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_, y_cont_, y_disc_)
                    D_fake, D_cont, D_disc = self.D(G_)
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(G_loss.data[0])

                    G_loss.backward(retain_graph=True)
                    self.G_optimizer.step()

                    self.train_hist['D_loss'].append(D_loss.data[0])
                    
                    # information loss
                    disc_loss = self.CE_loss(D_disc, torch.max(y_disc_, 1)[1])
                    cont_loss = self.MSE_loss(D_cont, y_cont_)
                    info_loss = disc_loss + cont_loss
                    self.train_hist['info_loss'].append(info_loss.data[0])
                    info_loss.backward()
                    self.info_optimizer.step()
                    
                    self.writer.add_scalar('info_loss', info_loss.data[0], step)

                # import IPython
                # IPython.embed()
                    
                self.writer.add_scalar('D_loss', D_loss.data[0], step)
                self.writer.add_scalar('D_Real_value', D_real_loss.data[0], step)
                self.writer.add_scalar('D_Fake_before_op', D_fake_loss.data[0], step)

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        
        self.writer.close()

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        samples = self.G(self.sample_z_, self.sample_c_, self.sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
