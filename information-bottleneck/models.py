import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.functional import softplus, softmax

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
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1

class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_classes, neg_slope=0.02, dropout=False, p_dropout=0.5):
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        
        self.layers = []
        self.num_neurons = [n_inputs] + n_hidden + [n_classes]
        self.best_performance = 0
        self.models = {}
        if dropout:
            print('Applying dropout with p=%s' % p_dropout)
        for i in range(len(self.num_neurons) - 2):
            self.layers.append(nn.Linear(self.num_neurons[i], self.num_neurons[i+1]))
            if dropout:
                self.layers.append(nn.Dropout(p_dropout))
            self.layers.append(nn.LeakyReLU(negative_slope=neg_slope))
            self.models['Linear{}'.format(i)] = nn.Sequential(*self.layers)

        self.layers.append(nn.Linear(self.num_neurons[i+1], self.num_neurons[i+2]))
        self.models['Output'] = nn.Sequential(*self.layers)
        self.full_model = self.models['Output']
        
        
    def forward(self, x, exitLayer=None): 
        if exitLayer is not None:
            out = self.models[exitLayer](x)
        else:
            out = self.full_model(x)
        return out


class VAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim=6):
        super(VAE, self).__init__()
        self.K = z_dim
        self.best_performance = 0

        self.encode = nn.Sequential(
            nn.Linear(x_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2*self.K))

        self.decode = nn.Sequential(
                nn.Linear(self.K, y_dim))

    def forward(self, x, num_sample=1):
        if x.dim() > 2 : x = x.view(x.size(0),-1)

        statistics = self.encode(x)
        mu = statistics[:,:self.K]
        std = softplus(statistics[:,self.K:]-5,beta=1)

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