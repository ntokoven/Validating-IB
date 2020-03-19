import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import argparse
import numpy as np
import pandas as pd
from random import randint, seed
import os
import matplotlib.pyplot as plt
import time

from data_utils import *
from evaluation import *
from helper import *
from models import VAE, MLP, MIEstimator


def train_encoder(dnn_hidden_units, dnn_input_units=784, dnn_output_units=10, z_dim = 6, enc_type='MLP', weight_decay=0, num_epochs=10, eval_freq=1):
    print('Weight decay to be applied: ', weight_decay)
    if enc_type == 'MLP':
        Net = MLP(dnn_input_units, dnn_hidden_units, dnn_output_units).to(device)
    elif enc_type =='VAE':
        Net = VAE(dnn_input_units, dnn_output_units, z_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Net.parameters(), lr=learning_rate, weight_decay=weight_decay) #default 1e-3

    start_time = time.time()
    max_accuracy = 0

    for epoch in range(num_epochs):
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.flatten(start_dim=1).to(device), y_train.to(device)
            optimizer.zero_grad()
            if enc_type =='VAE':
                (mu, std), out, z_train = Net(X_train)
                loss = criterion(out, y_train)
            else:
                out = Net(X_train)
                loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
        if epoch % eval_freq == 0 or epoch == num_epochs - 1:

                print('\n'+'#'*30)
                print('Training epoch - %d/%d' % (epoch+1, num_epochs))

                if enc_type == 'VAE':
                    (mu, std), out_test, z_test = Net(X_test)
                    test_loss = criterion(out_test, y_test)
                else:
                    out_test = Net(X_test)
                    test_loss = criterion(out_test, y_test)
                test_accuracy = accuracy(out_test, y_test)
                if test_accuracy > max_accuracy:
                    max_accuracy = test_accuracy

                print('Train: Accuracy - %0.3f, Loss - %0.3f' % (accuracy(out, y_train), loss))
                print('Test: Accuracy - %0.3f, Loss - %0.3f' % (test_accuracy, test_loss))
                print('Elapsed time: ', time.time() - start_time)
                print('#'*30,'\n')
                if test_accuracy == 1 and test_loss == 0:
                    break
    Net.best_performance = max_accuracy
    return Net


def train_MI(encoder, beta=0, mie_on_test=False, seed=69, num_epochs=2000, eval_freq=1, layer=''):
    
    if not mie_on_test:
        loader = train_loader
    else:
        loader = test_loader

    if enc_type == 'VAE':
       (_, _), _, z_test = encoder(X_test)
    else:
       z_test = encoder(X_test)
    z_test = torch.tensor(z_test, requires_grad=False).to(device)

    x_dim, y_dim, z_dim = X_test.shape[-1], dnn_output_units, z_test.shape[-1]

    mi_estimator_X = MIEstimator(x_dim, z_dim).to(device)
    mi_estimator_Y = MIEstimator(z_dim, y_dim).to(device)

    optimizer = optim.Adam([
    {'params': mi_estimator_X.parameters(), 'lr':mie_lr_x}, #default 1e-5
    {'params': mi_estimator_Y.parameters(), 'lr':mie_lr_y}, #default 1e-4
    ])
    if beta == 0:
        use_scheduler = True
        beta_scheduler = ExponentialScheduler(start_value=1e-6, end_value=1, n_iterations=500, start_iteration=20)
    else:
        use_scheduler = False

    start_time = time.time()
    max_MI_x, max_MI_y = 0, 0
    mi_mean_est_all = {'X': [], 'Y': []}
    
    for epoch in range(num_epochs):
        if use_scheduler:
            beta = beta_scheduler(epoch)
        mi_over_epoch = {'X': [], 'Y': []}
        for X, y in loader:
            y = onehot_encoding(y)
            X, y = X.flatten(start_dim=1).to(device), y.float().to(device)

            if enc_type == 'VAE':
                (_, _), _, z = encoder(X)
            else:
                z = encoder(X)

            optimizer.zero_grad()

            mi_gradient_X, mi_estimation_X = mi_estimator_X(X, z)
            mi_gradient_X = mi_gradient_X.mean()
            mi_estimation_X = mi_estimation_X.mean()

            mi_gradient_Y, mi_estimation_Y = mi_estimator_Y(z, y)
            mi_gradient_Y = mi_gradient_Y.mean()
            mi_estimation_Y = mi_estimation_Y.mean()
                    
            loss_mi = - mi_gradient_Y - beta * mi_gradient_X
            loss_mi.backward()
            optimizer.step()

            mi_over_epoch['X'].append(mi_estimation_X.item())
            mi_over_epoch['Y'].append(mi_estimation_Y.item())

        mi_over_epoch['X'] = np.array(mi_over_epoch['X'])
        mi_over_epoch['Y'] = np.array(mi_over_epoch['Y'])

        # Discard top and bottom 5% to avoid numerical outliers
        tmp = mi_over_epoch['X'][mi_over_epoch['X'] < np.quantile(mi_over_epoch['X'], 0.95)]
        tmp = tmp[tmp > np.quantile(mi_over_epoch['X'], 0.05)]
        mi_over_epoch['X'] = tmp

        tmp = mi_over_epoch['Y'][mi_over_epoch['Y'] < np.quantile(mi_over_epoch['Y'], 0.95)]
        tmp = tmp[tmp > np.quantile(mi_over_epoch['Y'], 0.05)]
        mi_over_epoch['Y'] = tmp

        if np.mean(mi_over_epoch['X']) > max_MI_x:
            max_MI_x = np.mean(mi_over_epoch['X'])
        if np.mean(mi_over_epoch['Y']) > max_MI_y:
            max_MI_y = np.mean(mi_over_epoch['Y'])
        mi_mean_est_all['X'].append(np.mean(mi_over_epoch['X']))
        mi_mean_est_all['Y'].append(np.mean(mi_over_epoch['Y']))
            
        if epoch % eval_freq == 0 or epoch == num_epochs - 1:

            print('#'*30)
            print('Step - ', epoch)
            print('Beta - ', beta)
                
            if epoch >= 10:
              delta_x = np.abs(mi_mean_est_all['X'][-2] - mi_mean_est_all['X'][-1])
              print('Delta X: ', delta_x)
              delta_y = np.abs(mi_mean_est_all['Y'][-2] - mi_mean_est_all['Y'][-1])
              print('Delta Y: ', delta_y)
              print('\nMean MI X for last 10', mi_mean_est_all['X'][-10:])
              print('Mean MI Y for last 10', mi_mean_est_all['Y'][-10:])
              mi_df = pd.DataFrame.from_dict(mi_mean_est_all)
              mi_df.to_csv('mie_%s_%s_%s_%s.csv' % (enc_type.lower(), 'test' if mie_on_test else 'train', beta if not use_scheduler else 'sched', int(1/weight_decay) if weight_decay != 0 else 0), sep=' ')
            print('Max I_est(X, Z) - %s' % max_MI_x)
            print('Max I_est(Z, Y) - %s' % max_MI_y)
            print('Elapsed time training MI for %s: %s' % (layer, time.time() - start_time))
            print('#'*30,'\n')

            if epoch >= 20 and np.mean(mi_mean_est_all['X'][-10]) > max_MI_x - 1e-1 or epoch == num_epochs - 1:
                plt.plot(np.arange(len(mi_df)), mi_df['X'], label='I(X,Z)')
                plt.plot(np.arange(len(mi_df)), mi_df['Y'], label='I(Z,Y)')
                plt.legend()
                plt.savefig('mie_curve_%s_%s_%s_%s_%s.png' % (enc_type.lower(), 'test' if mie_on_test else 'train', beta if not use_scheduler else 'sched', int(1/weight_decay) if weight_decay != 0 else 0, seed))
                break

            
    return max_MI_x, max_MI_y, mi_estimator_X, mi_estimator_Y

def build_information_plane(mi_values, layers_names, seeds):
    mi_df = pd.DataFrame.from_dict(mi_values)
    fig2, ax2 = plt.subplots(1, 1, sharex=True)
    colors = ['black', 'blue', 'red', 'green', 'yellow']
    for i in range(len(layers_names)):
        for j in range(len(seeds)):
            breakpoint()
            ax2.scatter(mi_df.loc[0, layers_names[i]][j], mi_df.loc[1, layers_names[i]][j], color=colors[i], label=layers_names[i])
            
            ax2.annotate(layers_names[i], (mi_df.loc[0, layers_names[i]][j], mi_df.loc[1, layers_names[i]][j]))
            ax2.grid()
    ax2.set_xlabel('I(X, Z)')
    ax2.set_ylabel('I(Z, Y)')

    fig2.legend()
    fig2.set_size_inches(10, 7, forward=True)
    fig2.savefig('info_plane_%s_%s_%s_%s.png' % (enc_type.lower(), 'test' if mie_on_test else 'train', mie_beta, int(1/weight_decay) if weight_decay != 0 else 0))

def main():

    Encoder = train_encoder(dnn_hidden_units, enc_type=enc_type, num_epochs=num_epochs, weight_decay=weight_decay)
    print('Best achieved performance: ', Encoder.best_performance)
    print(Encoder)

    if FLAGS.layers_to_track:
        layers_to_track = FLAGS.layers_to_track.split(",")
        layers_to_track = [int(layer_num) for layer_num in layers_to_track]
    else:
        layers_to_track = [1]

    pos_layers = - (np.array(layers_to_track) + 1)
    layers_names = []
    for pos in pos_layers:
        layers_names.append(list(get_named_layers(Encoder).keys())[pos].split('_')[0])


    if FLAGS.seeds:
        seeds = FLAGS.seeds.split(",")
        seeds = [int(seed) for seed in seeds]
    else:
        seeds = [default_seed]

    mie_layers = {} 
    start_time = time.time()

    for layer in layers_names:
        mie_layers[layer] = []

    for i in range(len(seeds)):

        print('\nRunning with seed %d out of %d' % (i+1, len(seeds)))
        torch.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        seed(seeds[i])
        
        
        for layer in layers_names:
            if enc_type == 'MLP':
                MI_X, MI_Y, MIE_X, MIE_Y = train_MI(Encoder.models[layer], beta=mie_beta, mie_on_test=mie_on_test, seed=seeds[i], layer=layer, num_epochs=mie_num_epochs)
            else:
                MI_X, MI_Y, MIE_X, MIE_Y = train_MI(Encoder, beta=mie_beta, mie_on_test=mie_on_test, seed=seeds[i], num_epochs=mie_num_epochs)
            if not os.path.exists(FLAGS.result_path):
                os.makedirs(FLAGS.result_path)
            torch.save(MIE_X.state_dict(), FLAGS.result_path + 'mie_x_%s_%s_%s_%s_%s.pt' % (enc_type.lower(), 'test' if mie_on_test else 'train', mie_beta, int(1/weight_decay) if weight_decay != 0 else 0, seed))
            torch.save(MIE_Y.state_dict(), FLAGS.result_path + 'mie_y_%s_%s_%s_%s_%s.pt' % (enc_type.lower(), 'test' if mie_on_test else 'train', mie_beta, int(1/weight_decay) if weight_decay != 0 else 0, seed))
            mie_layers[layer].append((MI_X, MI_Y))
            print('MI values for %s - %s, %s' % (layer, MI_X, MI_Y))
    
    build_information_plane(mie_layers, layers_names, seeds)
        
    print('Elapsed time - ', time.time() - start_time)


if __name__=='__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--enc_type', type = str, default = 'MLP',
                        help='Type of encoder to train')
    parser.add_argument('--dnn_hidden_units', type = str, default = '1024,512,256,128,64',
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--default_seed', type = int, default = 69,
                        help='Default seed for encoder training')
    parser.add_argument('--seeds', type = str, default = '9,42,103',
                        help='Comma separated list of random seeds')
    parser.add_argument('--layers_to_track', type = str, default = '1',
                        help='Comma separated list of inverse positions of encoding layers to evaluate (starting from 1)')
    parser.add_argument('--learning_rate', type = float, default = 1e-3,
                        help='Learning rate for encoder training')
    parser.add_argument('--mie_lr_x', type = float, default = 1e-5,
                        help='Learning rate for estimation of mutual information with input')
    parser.add_argument('--mie_lr_y', type = float, default = 1e-4,
                        help='Learning rate for estimation of mutual information with target')
    parser.add_argument('--mie_beta', type = float, default = 0,
                        help='Lagrangian multiplier representing prioirity of MI(z, y) over MI(x, z)')
    parser.add_argument('--mie_on_test', type = bool, default = False,
                        help='Whether to build MI estimator using training or test set')
    parser.add_argument('--weight_decay', type = float, default = 0,
                      help='Value of weight decay applied to optimizer')
    parser.add_argument('--num_epochs', type = int, default = 10,
                        help='Number of epochs to do training')
    parser.add_argument('--mie_num_epochs', type = int, default = 2000,
                        help='Max number of epochs to do MIE training')
    parser.add_argument('--num_classes', type = int, default = 10,
                        help='Number of classes')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=1,
                            help='Frequency of evaluation on the test set')
    parser.add_argument('--neg_slope', type=float, default=0.02,
                        help='Negative slope parameter for LeakyReLU')
    parser.add_argument('--result_path', type = str, default = 'results_mie',
                      help='Directory for storing results')

    FLAGS, unparsed = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available()

    default_seed = FLAGS.default_seed 
    num_classes = FLAGS.num_classes
    batch_size = FLAGS.batch_size 
    enc_type = FLAGS.enc_type
    weight_decay = FLAGS.weight_decay 
    num_epochs = FLAGS.num_epochs
    learning_rate = FLAGS.learning_rate
    mie_lr_x = FLAGS.mie_lr_x
    mie_lr_y = FLAGS.mie_lr_y
    mie_num_epochs = FLAGS.mie_num_epochs
    mie_beta = FLAGS.mie_beta
    mie_on_test = FLAGS.mie_on_test

    np.random.seed(default_seed)
    torch.manual_seed(default_seed)
    seed(default_seed)

    # Loading the MNIST dataset
    train_set = MNIST('./data/MNIST', download=True, train=True, transform=ToTensor())
    test_set = MNIST('./data/MNIST', download=True, train=False, transform=ToTensor())

    # Initialization of the data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)

    # Single time define the testing set. Keep it fixed until the end
    xs = []
    ys = []

    for x, y in test_loader:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs, 0)
    ys = torch.cat(ys, 0)

    X_test, y_test = torch.tensor(xs, requires_grad=False).flatten(start_dim=1).to(device), torch.tensor(ys, requires_grad=False).type(torch.LongTensor).to(device)

    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    dnn_input_units = X_test.shape[-1]
    dnn_output_units = num_classes

    main()