import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import argparse
import numpy as np
import pandas as pd
from random import randint, seed
from shapely.geometry import Polygon
import os
import matplotlib.pyplot as plt
import time

from data_utils import *
from evaluation import *
from helper import *
from models import VAE, MLP, MIEstimator
from train import train_encoder, train_encoder_VIB


def main():
    '''
    # functionality used to fine-tune the experiments without retraining encoders
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
    X_test, y_test = build_test_set(test_loader, device)
    if FLAGS.use_of_vib:
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

    if FLAGS.use_of_vib:
        z_test = Encoder(X_test)[-1].cpu().data.numpy()
        y_test = y_test.cpu().data.numpy()
    else:
        z_test = Encoder.models['Linear2'](X_test).cpu().data.numpy()
        y_test = y_test.cpu().data.numpy()
    

    # reduce dimensionality to 2D, we consider a subset of data because TSNE
    # is a slow algorithm
    
    # tsne_features = TSNE(n_components=2).fit_transform(z_test[:1000])
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(z_test)
    
    fig = plt.figure(figsize=(10, 6))

    # plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y_test[:tsne_features.shape[0]], marker='o',
                # edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
    
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=y_test[:pca_features.shape[0]], marker='o',
                edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
    plt.grid(False)
    plt.axis('off')
    plt.colorbar()
    fig.savefig(FLAGS.result_path+'/pca_manifold.png')
    
if __name__=='__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--enc_type', type = str, default = 'MLP',
                        help='Type of encoder to train')
    parser.add_argument('--p_dropout', type = float, default = 0,
                        help='Probability of dropout')
    parser.add_argument('--p_input_dropout', type = float, default = 0,
                        help='Probability of dropout')
    parser.add_argument('--encoder_hidden_units', type = str, default = '1024,1024,256',
                        help='Comma separated list of number of units in each hidden layer of encoder')
    parser.add_argument('--decoder_hidden_units', type = str, default = '',
                        help='Comma separated list of number of units in each hidden layer of decoder')
    parser.add_argument('--default_seed', type = int, default = 69,
                        help='Default seed for encoder training')
    parser.add_argument('--seeds', type = str, default = '69',
                        help='Comma separated list of random seeds')
    parser.add_argument('--num_seeds', type = int, default = 100,
                        help='If specified run for given amount of seed numbers in incremental setting')
    parser.add_argument('--layers_to_track', type = str, default = '_1',
                        help='Comma separated list of negative positions of encoding layers to evaluate with underscore as a minus sign (starting from _1:last before the classifying layer)')
    parser.add_argument('--learning_rate', type = float, default = 1e-3,
                        help='Learning rate for encoder training')
    parser.add_argument('--mie_lr_x', type = float, default = 3e-5,
                        help='Learning rate for estimation of mutual information with input')
    parser.add_argument('--mie_lr_y', type = float, default = 1e-4,
                        help='Learning rate for estimation of mutual information with target')
    parser.add_argument('--mie_beta', type = float, default = 1,
                        help='Lagrangian multiplier representing prioirity of MI(z, y) over MI(x, z)')
    parser.add_argument('--vib_beta', type = float, default = 1e-3,
                        help='Lagrangian multiplier representing prioirity of MI(z, y) over MI(x, z)')
    parser.add_argument('--use_of_vib', type = bool, default = False,
                        help='Need to train using Variational Information Bottleneck objective')
    parser.add_argument('--whiten_z', type = bool, default = False,
                        help='Need to normalize the distribution of latent variables before when building MIE')
    parser.add_argument('--mie_on_test', type = bool, default = False,
                        help='Whether to build MI estimator using training or test set')
    parser.add_argument('--mie_k_discard', type = float, default = 5,
                        help='Per cent of top and bottom MI estimations to discard')
    parser.add_argument('--mie_converg_bound', type = float, default = 5e-2,
                        help='Tightness of bound for the convergence criteria')
    parser.add_argument('--weight_decay', type = float, default = 0,
                      help='Value of weight decay applied to optimizer')
    parser.add_argument('--num_epochs', type = int, default = 20,
                        help='Number of epochs to do training')
    parser.add_argument('--mie_num_epochs', type = int, default = 100,
                        help='Max number of epochs to do MIE training')
    parser.add_argument('--mie_save_models', type = bool, default = False,
                      help='Need to store MIE models learnt')
    parser.add_argument('--mie_train_till_end', type = bool, default = False,
                      help='Need to train for mie_num_epochs or convergence')
    parser.add_argument('--num_classes', type = int, default = 10,
                        help='Number of classes')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=1,
                            help='Frequency of evaluation on the test set')
    parser.add_argument('--derive_w_size', type=int, default=500,
                            help='Compute the slope of the learning curve over this amount of training epochs')
    parser.add_argument('--w_size', type=int, default=20,
                            help='Window size to count towards convergence criteria')
    parser.add_argument('--neg_slope', type=float, default=0,
                        help='Negative slope parameter for LeakyReLU')
    parser.add_argument('--result_path', type = str, default = 'results_mie',
                      help='Directory for storing results')
    parser.add_argument('--comment', type = str, default = '',
                      help='Additional comments on the runtime set up')
    parser.add_argument('--use_pretrain', type = bool, default = False,
                      help='Need to load pretrained encoders or train from scratch')
    parser.add_argument('--mnist12k', type = bool,
                      help='Run for reduced MNIST 12k')
    
    FLAGS, unparsed = parser.parse_known_args()

    print_flags(FLAGS)

    global_start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available()

    default_seed = FLAGS.default_seed 
    num_classes = FLAGS.num_classes
    batch_size = FLAGS.batch_size 
    enc_type = FLAGS.enc_type
    p_dropout = FLAGS.p_dropout
    weight_decay = FLAGS.weight_decay 
    num_epochs = FLAGS.num_epochs
    learning_rate = FLAGS.learning_rate
    mie_lr_x = FLAGS.mie_lr_x
    mie_lr_y = FLAGS.mie_lr_y
    mie_num_epochs = FLAGS.mie_num_epochs
    mie_beta = FLAGS.mie_beta
    mie_on_test = FLAGS.mie_on_test
    mie_k_discard = FLAGS.mie_k_discard
    mie_train_till_end = FLAGS.mie_train_till_end
    mie_converg_bound = FLAGS.mie_converg_bound
    mie_save_models = FLAGS.mie_save_models
    w_size = FLAGS.w_size

    '''
    # If whitening helps adjust setting to apply also to weight_decay 0.
    # For now should manually set flag to True
    if weight_decay != 0:
        FLAGS.whiten_z = True
    '''

    if FLAGS.use_of_vib:
        enc_type = FLAGS.enc_type = 'VAE'

    if FLAGS.comment != '':
        FLAGS.result_path += '/%s' % FLAGS.comment

    np.random.seed(default_seed)
    torch.manual_seed(default_seed)
    seed(default_seed)

    # Loading the MNIST dataset
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
    else:

        train_set = MNIST('./data/MNIST', download=True, train=True, transform=ToTensor())
        test_set = MNIST('./data/MNIST', download=True, train=False, transform=ToTensor())

    # Initialization of the data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)

    # Single time define the testing set. Keep it fixed until the end
    X_test, y_test = build_test_set(test_loader, device)

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
    
    dnn_input_units = X_test.shape[-1]
    dnn_output_units = num_classes

    main()

    print('Excecution finished with overall time elapsed - %s \n\n' % (time.time() - global_start_time))
    print_flags(FLAGS)