import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.functional import softplus, softmax
import pandas as pd
from numbers import Number

# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()
        
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )
    
    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1)) #Positive Samples 
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1)) #Predictions for shuffled (negative) samples from p(z1)p(z2)
        # return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1 # I_NWJ, I_JS
        return -softplus(-pos) - softplus(neg), pos - neg.exp() + 1 # I_NWJ, I_JS


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Deterministic(nn.Module):
    
    def __init__(self, x_dim, z_dims, dec_dims, y_dim, FLAGS):
        #self.n_channels, self.n_classes = 3, 10
        super(Deterministic, self).__init__()

        p_dropout, neg_slope = FLAGS.p_dropout, FLAGS.neg_slope
        self.K = z_dims[-1]
        self.layers = []
        self.enc_num_neurons = [x_dim] + z_dims 
        self.dec_num_neurons = [self.K] + dec_dims + [y_dim]
        self.best_performance = 0
        self.models = {}

        if FLAGS.cifar10:
            self.encode = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1), 
                nn.ReLU(),
                # nn.BatchNorm2d(64), 
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Dropout(FLAGS.p_dropout),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(128), 
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Dropout(FLAGS.p_dropout),
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(256), 
                nn.Dropout(FLAGS.p_dropout),
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(256), 
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Dropout(FLAGS.p_dropout),
                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(512), 
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.Dropout(FLAGS.p_dropout),
                nn.ReLU(),
                # nn.BatchNorm2d(512), 
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Dropout(FLAGS.p_dropout),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(512), 
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(512), 
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Dropout(FLAGS.p_dropout),
                nn.Flatten(),
                nn.Linear(512, self.K))
        else:
            for i in range(len(self.enc_num_neurons) - 2):
                self.layers.append(nn.Linear(self.enc_num_neurons[i], self.enc_num_neurons[i+1]))
                self.layers.append(nn.LeakyReLU(negative_slope=FLAGS.neg_slope))
                if FLAGS.p_dropout != 0:
                    self.layers.append(nn.Dropout(FLAGS.p_dropout))
                self.models['Linear{}'.format(i)] = nn.Sequential(*self.layers)
            self.layers.append(nn.Linear(self.enc_num_neurons[i+1], self.K))
            self.models['Linear{}'.format(i+1)] = nn.Sequential(*self.layers)
            self.encode = self.models['Linear{}'.format(i+1)]
        
        print('Encoder architecture', self.encode)

        self.dec_num_neurons = [self.K] + dec_dims + [y_dim]
        self.dec_layers = []
        for i in range(len(self.dec_num_neurons) - 2):
            self.dec_layers.append(nn.Linear(self.dec_num_neurons[i], self.dec_num_neurons[i+1]))
            self.dec_layers.append(nn.LeakyReLU(negative_slope=FLAGS.neg_slope))
            if FLAGS.p_dropout != 0:
                self.dec_layers.append(nn.Dropout(FLAGS.p_dropout))
        self.dec_layers.append(nn.Linear(self.dec_num_neurons[-2], y_dim))
        
        self.decode = nn.Sequential(*self.dec_layers)

    def forward(self, x): 
        out = self.decode(self.encode(x))
        return out
        

class MLP(nn.Module):
    def __init__(self, x_dim, z_dims, dec_dims, y_dim, FLAGS):
        super(MLP, self).__init__()

        self.K = z_dims[-1]
        self.layers = []
        self.enc_num_neurons = [x_dim] + z_dims 
        self.dec_num_neurons = [self.K] + dec_dims + [y_dim]
        self.best_performance = 0
        self.models = {}

        for i in range(len(self.enc_num_neurons) - 2):
            self.layers.append(nn.Linear(self.enc_num_neurons[i], self.enc_num_neurons[i+1]))
            self.layers.append(nn.LeakyReLU(negative_slope=FLAGS.neg_slope))
            if FLAGS.p_dropout != 0:
                self.layers.append(nn.Dropout(FLAGS.p_dropout))
            self.models['Linear{}'.format(i)] = nn.Sequential(*self.layers)
        self.layers.append(nn.Linear(self.enc_num_neurons[i+1], self.K))
        self.models['Linear{}'.format(i+1)] = nn.Sequential(*self.layers)
        self.encode = self.models['Linear{}'.format(i+1)]
        
        self.dec_layers = []
        for i in range(len(self.dec_num_neurons) - 2):
            self.dec_layers.append(nn.Linear(self.dec_num_neurons[i], self.dec_num_neurons[i+1]))
            self.dec_layers.append(nn.LeakyReLU(negative_slope=FLAGS.neg_slope))
            if FLAGS.p_dropout != 0:
                self.dec_layers.append(nn.Dropout(FLAGS.p_dropout))
            self.models['Linear{}'.format(i)] = nn.Sequential(*self.layers)
        
        self.dec_layers.append(nn.Linear(self.dec_num_neurons[-2], y_dim))
        
        self.decode = nn.Sequential(*self.dec_layers)
        
        
    def forward(self, x): 
        out = self.decode(self.encode(x))
        return out

class Stochastic(nn.Module):
    def __init__(self, x_dim, z_dims, dec_dims, y_dim, FLAGS):
        super(Stochastic, self).__init__()

        self.CEB = FLAGS.use_of_ceb
        self.K = z_dims[-1]
        self.best_performance = 0

        self.num_neurons = [x_dim] + z_dims
        if FLAGS.cifar10:
            print('Building VAE with Convolutional architecture to work with CIFAR10')
            #VGG11
            self.encode = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1), 
                # nn.BatchNorm2d(64), 
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                # nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                # nn.BatchNorm2d(256), 
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                # nn.BatchNorm2d(256), 
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                # nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                # nn.BatchNorm2d(512), 
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                # nn.BatchNorm2d(512), 
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                # nn.BatchNorm2d(512), 
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Flatten(),
                nn.Linear(512, 2 * self.K))
        else:
            self.layers = []
            for i in range(len(self.num_neurons) - 2):
                self.layers.append(nn.Linear(self.num_neurons[i], self.num_neurons[i+1]))
                self.layers.append(nn.LeakyReLU(negative_slope=FLAGS.neg_slope))
                if FLAGS.p_dropout != 0:
                    self.layers.append(nn.Dropout(FLAGS.p_dropout))
            self.layers.append(nn.Linear(self.num_neurons[i+1], 2 * self.K))

            self.encode = nn.Sequential(*self.layers)
            
        self.dec_num_neurons = [self.K] + dec_dims + [y_dim]
        self.dec_layers = []
        for i in range(len(self.dec_num_neurons) - 2):
            self.dec_layers.append(nn.Linear(self.dec_num_neurons[i], self.dec_num_neurons[i+1]))
            self.dec_layers.append(nn.LeakyReLU(negative_slope=FLAGS.neg_slope))
            if FLAGS.p_dropout != 0:
                self.dec_layers.append(nn.Dropout(FLAGS.p_dropout))
        self.dec_layers.append(nn.Linear(self.dec_num_neurons[-2], y_dim))
        
        self.decode = nn.Sequential(*self.dec_layers)

    def forward(self, x, num_sample=1):
        # if x.dim() > 2 : x = x.view(x.size(0),-1) # commented when deploying VGG architecture

        statistics = self.encode(x)
        mu = statistics[:,:self.K]
        std = softplus(statistics[:,self.K:]-5,beta=1) + 1e-7

        encoding = self.reparametrize_n(mu, std, num_sample)
        logit = self.decode(encoding)

        if num_sample == 1 : pass
        elif num_sample > 1 : logit = softmax(logit, dim=2).mean(0)

        return (mu, std), logit, encoding

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()


            