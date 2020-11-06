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
import time, datetime


from data_utils import *
from evaluation import *
from helper import *
from option import get_option

from models import Stochastic, Deterministic, MLP, MIEstimator
from train import train_encoder, train_encoder_VIB

import umap.umap_ as umap


def get_info(FLAGS, bound=False):
    info = 'bound_' if bound else 'model_'
    if FLAGS.use_of_ceb:
        model_type = 'CVIB'
        value = FLAGS.vib_beta
    elif FLAGS.use_of_vib:
        model_type = 'VIB'
        value = FLAGS.vib_beta
    return info + model_type + '_' + str(value).replace('.', '') + ('_' + FLAGS.comment if FLAGS.comment else '')


def main():

    if FLAGS.cifar10:
        X_test, y_test = build_test_set(test_loader, device, flatten=False)
    else:
        X_test, y_test = build_test_set(test_loader, device)

    
    # '''
    # functionality used to fine-tune the experiments without retraining encoders
    if True: #FLAGS.use_pretrain and os.path.exists('pretrained_encoders/enc_%s_%sepochs.pt' % ('ceb' if FLAGS.use_of_ceb else 'vib', FLAGS.num_epochs)):
        print('\n\nLoading pretrained by %s for %s epochs' % ('ceb' if FLAGS.use_of_ceb else 'vib', FLAGS.num_epochs))

        dnn_input_units = X_test.shape[1] if FLAGS.cifar10 else X_test.shape[-1]
        
        dnn_output_units = FLAGS.num_classes
        if FLAGS.enc_type == 'stoch':
            print('\nBuilding stochastic encoder')
            Encoder = Stochastic(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device) 
        elif FLAGS.enc_type == 'determ':
            print('\nBuilding deterministic encoder')
            Encoder = Deterministic(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device)
        else:
            print('\nBuilding Vanilla MLP')
            Encoder = MLP(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device)
        Encoder.load_state_dict(torch.load(FLAGS.path_to_encoder_model))#'pretrained_encoders/enc_%s_%sepochs.pt' % ('ceb' if FLAGS.use_of_ceb else 'vib', FLAGS.num_epochs)))
        print(Encoder)
        breakpoint()
    else:
        if FLAGS.use_of_vib or FLAGS.use_of_ceb:
            Encoder, info_bound = train_encoder_VIB(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device)
            if not os.path.exists(FLAGS.result_path):
                os.makedirs(FLAGS.result_path)
            with open(FLAGS.result_path+'/%s.txt' % get_info(FLAGS, bound=True), 'w') as f:
                f.write(str(info_bound))
        else:
            Encoder = train_encoder(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device)
        if FLAGS.save_encoder:
            if not os.path.exists(FLAGS.result_path+'/fixed_encoders'):
                os.makedirs(FLAGS.result_path+'/fixed_encoders')
            
            # torch.save(Encoder.state_dict(), 'pretrained_encoders/enc_%s_%sepochs.pt' % ('ceb' if FLAGS.use_of_ceb else 'vib', FLAGS.num_epochs))
            # print('Saved encoder that has been trained for %s epochs' % FLAGS.num_epochs)

            torch.save(Encoder.state_dict(), FLAGS.result_path+'/fixed_encoders/%s.pt' % (get_info(FLAGS)))
            print('Storing %s' % get_info(FLAGS))


    '''
    if FLAGS.use_of_vib:
        z_test = Encoder(X_test)[-1].cpu().data.numpy()
        y_test = y_test.cpu().data.numpy()
    '''


if __name__=='__main__':
    # Command line arguments

    parser = argparse.ArgumentParser()
    FLAGS, unparsed = get_option(parser)

    global_start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    cuda = torch.cuda.is_available()

    if FLAGS.dataset == 'cifar10':
        FLAGS.cifar10 = True
    elif FLAGS.dataset == 'mnist12k':
        FLAGS.mnist12k = True
    
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

    # Loading the MNIST dataset
    if FLAGS.cifar10:
        print('Uploading CIFAR10')
        # cifar10 = cifar10_utils.get_cifar10('data/cifar10/cifar-10-batches-py')
        # X_test, y_test = torch.tensor(cifar10['test'].images, requires_grad=False).to(device), torch.tensor(cifar10['test'].labels, requires_grad=False).to(device)
        
        transform = Compose(
                    [ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # train_set = CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
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
    elif not FLAGS.cifar10:
        print('Uploading Regular MNIST')
        train_set = MNIST('./data/MNIST', download=True, train=True, transform=ToTensor())
        test_set = MNIST('./data/MNIST', download=True, train=False, transform=ToTensor())

    # Single time define the testing set. Keep it fixed until the end
    # if not FLAGS.cifar10:
    #  Initialization of the data loader

    # train_loader = DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1)

    # else:
        # train_loader, test_loader = None, None
        
    # TODO: change to work properly with Convolutional architecture
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