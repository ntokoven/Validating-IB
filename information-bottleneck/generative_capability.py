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
    
    if FLAGS.use_of_vib:
        Encoder = train_encoder_VIB(FLAGS, dnn_hidden_units, train_loader, test_loader, device)
    else:
        Encoder = train_encoder(FLAGS, dnn_hidden_units, train_loader, test_loader, device)
    print('Best achieved performance: %s \n' % Encoder.best_performance)
    print(Encoder)

    if FLAGS.layers_to_track:
        layers_to_track = FLAGS.layers_to_track.replace('_', '-').split(",")
        layers_to_track = [int(layer_num) for layer_num in layers_to_track]
    else:
        layers_to_track = [-1]

    pos_layers = np.array(layers_to_track) - 1
    layers_names = []
    for pos in pos_layers:
        layers_names.append(list(get_named_layers(Encoder).keys())[pos].split('_')[0])

    if FLAGS.seeds:
        seeds = FLAGS.seeds.split(",")
        seeds = [int(seed) for seed in seeds]
    else:
        seeds = [default_seed]

    if FLAGS.num_seeds:
        seeds = np.arange(1, FLAGS.num_seeds + 1)
    print('Seeds to evaluate - ', seeds)
    print('Performing custom split to get training subsets with different amount of labeled examples')
    train_subsets = build_training_subsets(train_set, base=2, num_classes=num_classes)
    num_labels_range = list(train_subsets.keys()) 
    print('Done custom split \n')

    acc = {layer:{i:[] for i in num_labels_range} for layer in layers_names}

    start_time = time.time()

    for i in range(len(seeds)):

        print('\nRunning with seed %d out of %d' % (i+1, len(seeds)))
        torch.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        seed(seeds[i])

        for layer in layers_names:
            for num_labels in num_labels_range:
                print('\n\nEvaluating for %s (labels per class - %d)' % (layer, int(num_labels/num_classes)))
                if enc_type == 'MLP':
                    train_accuracy, test_accuracy = evaluate(encoder=Encoder.models[layer], enc_type=enc_type, train_on=train_subsets[num_labels], test_on=test_set, cuda=torch.cuda.is_available())
                else:
                    train_accuracy, test_accuracy = evaluate(encoder=Encoder, enc_type=enc_type, train_on=train_subsets[num_labels], test_on=test_set, cuda=torch.cuda.is_available())

                print('Train Accuracy: %f'% train_accuracy)
                print('Test Accuracy: %f'% test_accuracy)
                
                acc[layer][num_labels].append(test_accuracy)
        print('Elapsed time - ', time.time() - start_time)

    acc_df = pd.DataFrame.from_dict(acc)
    if not os.path.exists(FLAGS.result_path):
          os.makedirs(FLAGS.result_path)
    acc_df.to_csv(FLAGS.result_path+'/acc_%s_%s.csv' % (enc_type.lower(), int(1/weight_decay) if weight_decay != 0 else 0), sep=' ')
    plot_acc_numlabels(FLAGS, acc_df, layers_names, Encoder.best_performance, num_labels_range)


if __name__=='__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--enc_type', type = str, default = 'MLP',
                        help='Type of encoder to train')
    parser.add_argument('--p_dropout', type = float, default = 0,
                        help='Probability of dropout')
    parser.add_argument('--p_input_dropout', type = float, default = 0,
                        help='Probability of dropout')
    parser.add_argument('--dnn_hidden_units', type = str, default = '1024,512,256,128,64',
                        help='Comma separated list of number of units in each hidden layer')
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
    train_set = MNIST('./data/MNIST', download=True, train=True, transform=ToTensor())
    test_set = MNIST('./data/MNIST', download=True, train=False, transform=ToTensor())

    # Initialization of the data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)

    # Single time define the testing set. Keep it fixed until the end
    X_test, y_test = build_test_set(test_loader, device)

    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    dnn_input_units = X_test.shape[-1]
    dnn_output_units = num_classes

    main()

    print('Excecution finished with overall time elapsed - %s \n\n' % (time.time() - global_start_time))
    print_flags(FLAGS)