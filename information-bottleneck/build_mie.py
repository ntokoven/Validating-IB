import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

import argparse
import numpy as np
import pandas as pd
from random import randint, seed
import os
import matplotlib.pyplot as plt
import math
import time, datetime

from data_utils import *
from evaluation import *
from helper import *
from option import get_option
from models import Stochastic, Deterministic, MLP, MIEstimator
from train import train_encoder, train_encoder_VIB

def train_MI(encoder, beta=1, mie_on_test=False, seed=69, num_epochs=2000, layer=''):
    
    def prun_quantile(x, k):
        x_prun = x[x < np.quantile(x, 1 - k/100)]
        x_prun = x_prun[x_prun > np.quantile(x, k/100)]
        return x_prun

    if not mie_on_test:
        loader = train_loader
    else:
        loader = test_loader
    # Keep full training set for the final MI estimations (validation set)
    if FLAGS.cifar10:
        X_test, y_test = build_test_set(train_loader, device, flatten=False, each=5)
    elif FLAGS.mnist12k:
        X_test, y_test = build_test_set(train_loader, device, each=1)
    else:
        X_test, y_test = build_test_set(train_loader, device, each=5)
    
    with torch.no_grad():
        if FLAGS.enc_type == 'stoch':
            (_, _), _, z_test = encoder(X_test)
            z_test = z_test.detach().to(device)
        else:
            z_test = encoder(X_test).detach().to(device)

    x_dim, y_dim, z_dim = X_test.flatten(start_dim=1).shape[-1], FLAGS.num_classes, z_test.shape[-1]
    mi_estimator_X = MIEstimator(x_dim, z_dim).to(device)
    mi_estimator_Y = MIEstimator(z_dim, y_dim).to(device)

    optimizer = optim.Adam([
    {'params': mi_estimator_X.parameters(), 'lr': FLAGS.mie_lr_x}, #default 3e-5
    {'params': mi_estimator_Y.parameters(), 'lr': FLAGS.mie_lr_y}, #default 1e-4
    ])

    start_time = time.time()
    max_MI_x = max_MI_y = 0
    train_x = train_y = True
    mi_mean_est_all = {'X': [], 'Y': []}
    
    for epoch in range(num_epochs):
        mi_over_epoch = {'X': [], 'Y': []}
        for X, y in loader:
            y = onehot_encoding(y)

            if not FLAGS.cifar10:
                X, y = X.flatten(start_dim=1).to(device), y.float().to(device)
            else: 
                X, y = X.float().to(device), y.float().to(device)
            
            if FLAGS.enc_type == 'stoch':
                (_, _), _, z = encoder(X.float())
            else:
                z = encoder(X)

            if FLAGS.whiten_z:
                z = whitening(z)

            if FLAGS.cifar10:
                X = X.flatten(start_dim=1)

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
            
            del(X)
            del(z)

        with torch.no_grad():
            mi_over_epoch['X'] = np.nan_to_num(np.array(mi_over_epoch['X']))
            mi_over_epoch['Y'] = np.nan_to_num(np.array(mi_over_epoch['Y']))
            
            # Discard top and bottom k% to avoid numerical outliers
            mi_over_epoch['X'] = prun_quantile(mi_over_epoch['X'], FLAGS.mie_k_discard)
            mi_over_epoch['Y'] = prun_quantile(mi_over_epoch['Y'], FLAGS.mie_k_discard)


            if np.mean(mi_over_epoch['X']) > max_MI_x:
                max_MI_x = np.mean(mi_over_epoch['X'])
            if np.mean(mi_over_epoch['Y']) > max_MI_y:
                max_MI_y = np.mean(mi_over_epoch['Y'])
            mi_mean_est_all['X'].append(np.mean(mi_over_epoch['X']))
            mi_mean_est_all['Y'].append(np.mean(mi_over_epoch['Y']))
                
            if epoch % FLAGS.eval_freq == 0 or epoch == num_epochs - 1:

                print('#'*30)
                print('Step - ', epoch)
                    
                if epoch >= 2:
                    delta_x = mi_mean_est_all['X'][-2] - mi_mean_est_all['X'][-1]
                    print('Delta X: ', delta_x)
                    delta_y = mi_mean_est_all['Y'][-2] - mi_mean_est_all['Y'][-1]
                    print('Delta Y: ', delta_y)
                if epoch >= 10:
                    print('\nMean MI X for last 10', np.mean(mi_mean_est_all['X'][-10:]))
                    print('Mean MI Y for last 10', np.mean(mi_mean_est_all['Y'][-10:]))
                if epoch >= 20:
                    print('\nMean MI X for last 20', np.mean(mi_mean_est_all['X'][-20:]))
                    print('Mean MI Y for last 20', np.mean(mi_mean_est_all['Y'][-20:]))
                if epoch >= 30:
                    print('\nMean MI X for last 30', np.mean(mi_mean_est_all['X'][-30:]))
                    print('Mean MI Y for last 30', np.mean(mi_mean_est_all['Y'][-30:]))
                if epoch >= FLAGS.derive_w_size+1:
                    # Measuring the derivative over derive_w_size last epochs 
                    # delta_f(x_i) = (f(x) - f(x_i)) / (x - x_i)
                    print('d I(X;Z) / d epoch:', (mi_mean_est_all['X'][-(FLAGS.derive_w_size+1)] - mi_mean_est_all['X'][-1]) / FLAGS.derive_w_size)
                    print('d I(Y;Z) / d epoch:', (mi_mean_est_all['Y'][-(FLAGS.derive_w_size+1)] - mi_mean_est_all['Y'][-1]) / FLAGS.derive_w_size)
                if epoch >= 2 * FLAGS.w_size:
                    print('Latest window mean value: ', np.mean(mi_mean_est_all['X'][-FLAGS.w_size:]))
                    print('Previous window mean value', np.mean(mi_mean_est_all['X'][-2*FLAGS.w_size:-FLAGS.w_size]))
                print('Max I_est(X, Z) - %s' % max_MI_x)
                print('Max I_est(Z, Y) - %s' % max_MI_y)
                print('Elapsed time training MI for %s: %s' % (layer, time.time() - start_time))
                print('#'*30,'\n')
                
                mi_df = pd.DataFrame.from_dict(mi_mean_est_all)
                if not os.path.exists(FLAGS.result_path+'/mie_train_values'):
                    os.makedirs(FLAGS.result_path+'/mie_train_values')
                mi_df.to_csv(FLAGS.result_path+'/mie_train_values/mie_%s_%s_l%s_s%s.csv' % (FLAGS.enc_type.lower(), 'test' if mie_on_test else 'train', layer, seed), sep=' ')
                
                plot_mie_curve(FLAGS, mi_df, layer, seed)
            
            if epoch >= 500 and max_MI_x < 1e-1:
                print('No Mutual Information preserved by the model')
                break
            if epoch >= FLAGS.w_size and np.mean(mi_mean_est_all['X'][-2 * FLAGS.w_size:-FLAGS.w_size]) > np.mean(mi_mean_est_all['X'][-FLAGS.w_size:]) - FLAGS.mie_converg_bound:
                train_x = False

            if epoch >= FLAGS.w_size and np.mean(mi_mean_est_all['Y'][-2 * FLAGS.w_size:-FLAGS.w_size]) > np.mean(mi_mean_est_all['Y'][-FLAGS.w_size:]) - FLAGS.mie_converg_bound:
                train_y = False
            if not FLAGS.mie_train_till_end:
                if train_x == False and train_y == False:
                    print('Convergence criteria successfully satisified.\n\n')
                    break
            

    plot_mie_curve(FLAGS, mi_df, layer, seed)  

    with torch.no_grad():
        if FLAGS.enc_type == 'stoch':
            (_, _), _, z_test = encoder(X_test)
            z_test = z_test.detach().to(device)
        else:
            z_test = encoder(X_test).detach().to(device)
        
        X_test = X_test.flatten(start_dim=1).to(device)
        y_test = onehot_encoding(y_test, device=device).float()

        mi_gradient_X_final, mi_estimation_X_final = mi_estimator_X(X_test, z_test)
        mi_estimation_X_final = np.nan_to_num(mi_estimation_X_final.cpu().data.numpy())
        mi_estimation_X_final = prun_quantile(mi_estimation_X_final, FLAGS.mie_k_discard)

        mi_gradient_Y_final, mi_estimation_Y_final = mi_estimator_Y(z_test, y_test)
        mi_estimation_Y_final = np.nan_to_num(mi_estimation_Y_final.cpu().data.numpy())
        mi_estimation_Y_final = prun_quantile(mi_estimation_Y_final, FLAGS.mie_k_discard)
        print('ELEMENTWISE ESTIMATIONS ', mi_estimation_X_final, mi_estimation_Y_final)

        mi_estimation_X_final = mi_estimation_X_final.mean()
        mi_estimation_Y_final = mi_estimation_Y_final.mean()

        print('\nFINAL', mi_estimation_X_final, mi_estimation_Y_final)


    return mi_estimation_X_final, mi_estimation_Y_final, mi_estimator_X, mi_estimator_Y

def main():
    '''
    # For debugging purposes
    if FLAGS.use_pretrain and os.path.exists('pretrained_encoders/enc_%sepochs.pt' % FLAGS.num_epochs):
        Encoder = MLP(dnn_input_units, dnn_hidden_units, dnn_output_units, FLAGS).to(device)
        Encoder.load_state_dict(torch.load('pretrained_encoders/enc_%sepochs.pt' % FLAGS.num_epochs))
    else:
        if FLAGS.use_of_vib:
            Encoder = train_encoder_VIB(FLAGS, dnn_hidden_units, train_loader, test_loader, device)
        else:
            Encoder = train_encoder(FLAGS, dnn_hidden_units, train_loader, test_loader, device)
        torch.save(Encoder.state_dict(), 'pretrained_encoders/enc_%sepochs.pt' % FLAGS.num_epochs)
        print('Saved encoder that has been trained for %s epochs' % FLAGS.num_epochs)
    '''
    if FLAGS.use_of_vib or FLAGS.use_of_ceb:
        Encoder = train_encoder_VIB(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device)
    else:
        Encoder = train_encoder(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device)
    print('Best achieved performance: %s \n' % Encoder.best_performance)
    # '''
    print(Encoder)

    if FLAGS.layers_to_track:
        layers_to_track = FLAGS.layers_to_track.replace('_', '-').split(",")
        layers_to_track = [int(layer_num) - len(decoder_hidden_units) for layer_num in layers_to_track]
    else:
        layers_to_track = [-(1+len(decoder_hidden_units))]

    pos_layers = np.array(layers_to_track) - 1
    layers_names = []
    for pos in pos_layers:
        layers_names.append(list(get_named_layers(Encoder).keys())[pos].split('_')[0])

    if FLAGS.seeds:
        seeds = FLAGS.seeds.split(",")
        seeds = [int(seed) for seed in seeds]
    else:
        seeds = [FLAGS.default_seed]

    mie_layers = {layer:{s:(np.nan, np.nan) for s in seeds} for layer in layers_names}
    avg_max_mie_df = pd.Series(dtype=float)
    get_x, get_y = lambda x: x[0], lambda x: x[1]

    start_time = time.time()
        
    
    for i in range(len(seeds)):

        print('\nRunning for seed %d out of %d' % (i+1, len(seeds)))
        torch.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        seed(seeds[i])
        
        for j in range(len(layers_names)):
            layer = layers_names[j]
            if FLAGS.enc_type == 'MLP':
                MI_X, MI_Y, MIE_X, MIE_Y = train_MI(Encoder.models[layer], mie_on_test=FLAGS.mie_on_test, seed=seeds[i], layer=layer, num_epochs=FLAGS.mie_num_epochs)
            else:
                MI_X, MI_Y, MIE_X, MIE_Y = train_MI(Encoder, mie_on_test=FLAGS.mie_on_test, seed=seeds[i], num_epochs=FLAGS.mie_num_epochs)
            
            mie_layers[layer][seeds[i]] = (MI_X, MI_Y)
            max_mie_df = pd.DataFrame.from_dict(mie_layers).transpose()
            print('max_mie_df\n', max_mie_df)
            if not os.path.exists(FLAGS.result_path+'/mie_max_values'):
                os.makedirs(FLAGS.result_path+'/mie_max_values')

            for name in layers_names:
                avg_max_mie_df.loc[name] = (max_mie_df.loc[name].apply(get_x).mean(), max_mie_df.loc[name].apply(get_y).mean())
            avg_max_mie_df.to_csv(FLAGS.result_path+'/mie_max_values/layers_%s_seeds_%s.csv' % (concat(layers_names), concat(seeds)), sep=' ')
            print('avg_max_values\n', avg_max_mie_df)
            print('MI values for %s - %s, %s' % (layer, MI_X, MI_Y))
            build_information_plane(mie_layers, layers_names[:j+1], seeds[:i+1], FLAGS)

            #Saving models for reproducibility
            if FLAGS.mie_save_models:
                if not os.path.exists(FLAGS.result_path+'/estimator_models'):
                    os.makedirs(FLAGS.result_path+'/estimator_models')
                torch.save(MIE_X.state_dict(), FLAGS.result_path + '/estimator_models/mie_x_%s_%s_l%s_s%s.pt' % (FLAGS.enc_type.lower(), 'test' if FLAGS.mie_on_test else 'train', layer, seeds[i]))
                torch.save(MIE_Y.state_dict(), FLAGS.result_path + '/estimator_models/mie_y_%s_%s_l%s_s%s.pt' % (FLAGS.enc_type.lower(), 'test' if FLAGS.mie_on_test else 'train', layer, seeds[i]))
                print('Saved models successfully ')
                print(FLAGS.result_path+'/estimator_models')
        
    print('Elapsed time - ', time.time() - start_time)


if __name__=='__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = get_option(parser)

    global_start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available()
    
    if FLAGS.dataset == 'cifar10':
        FLAGS.cifar10 = True
    elif FLAGS.dataset == 'mnist12k':
        FLAGS.mnist12k = True

    # If whitening helps adjust setting to apply also to weight_decay 0.
    # For now should manually set flag to True
    if FLAGS.weight_decay != 0:
        print('MIE evaluation should be done for the normalized distribution')
        FLAGS.whiten_z = True

    if FLAGS.use_of_vib or FLAGS.use_of_ceb:
        FLAGS.enc_type = 'stoch'
    
    if FLAGS.enc_type != 'stoch':
        print('Should be here')
        FLAGS.enc_type = 'determ'

    if FLAGS.comment == 'unit':
        FLAGS.unit_sigma = True

    if FLAGS.comment != '':
        FLAGS.result_path += '/%s' % FLAGS.comment

    print_flags(FLAGS)

    np.random.seed(FLAGS.default_seed)
    torch.manual_seed(FLAGS.default_seed)
    seed(FLAGS.default_seed)

    if FLAGS.cifar10:
        print('Uploading CIFAR10')
        transform = Compose(
                    [ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
        test_set = CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)
    if FLAGS.mnist12k:
        print('Uploading MNIST 12k')
        train_data = np.loadtxt('data/mnist12k/mnist_train.amat')
        X_train = train_data[:, :-1] / 1.0
        y_train = train_data[:, -1:]

        test_data = np.loadtxt('data/mnist12k/mnist_test.amat')
        X_test = test_data[:, :-1] / 1.0
        y_test = test_data[:, -1:]

        X_train, y_train, X_test, y_test = torch.FloatTensor(X_train), torch.LongTensor(y_train), torch.FloatTensor(X_test), torch.LongTensor(y_test)
        train_set = torch.utils.data.TensorDataset(X_train, y_train)
        test_set = torch.utils.data.TensorDataset(X_test, y_test)
        print('Done MNIST 12k')
    # Loading the MNIST dataset
    elif not FLAGS.cifar10:
        print('Uploading Regular MNIST')
        train_set = MNIST('./data/MNIST', download=True, train=True, transform=ToTensor())
        test_set = MNIST('./data/MNIST', download=True, train=False, transform=ToTensor())

    train_loader = DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1)
        
    if FLAGS.encoder_hidden_units:
        encoder_hidden_units = FLAGS.encoder_hidden_units.split(",")
        encoder_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in encoder_hidden_units]
    else:
        encoder_hidden_units = []
    if FLAGS.decoder_hidden_units:
        decoder_hidden_units = FLAGS.decoder_hidden_units.split(",")
        decoder_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in decoder_hidden_units]
    else:
        decoder_hidden_units = []
    
    main()

    print_flags(FLAGS)
    print('Excecution finished with overall time elapsed - %s \n\n' % str(datetime.timedelta(seconds=int(time.time() - global_start_time))))