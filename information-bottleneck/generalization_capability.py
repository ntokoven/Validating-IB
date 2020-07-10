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
from shapely.geometry import Polygon
import os
import matplotlib.pyplot as plt
import time, datetime
import cifar10_utils

from data_utils import *
from evaluation import *
from helper import *
from option import get_option

from models import VAE, MLP, MIEstimator
from train import train_encoder, train_encoder_VIB


def main():
    # '''
    # functionality used to fine-tune the experiments without retraining encoders
    if FLAGS.use_pretrain and os.path.exists('pretrained_encoders/enc_%s_%sepochs.pt' % ('ceb' if FLAGS.use_of_ceb else 'vib', FLAGS.num_epochs)):
        print('Loading pretrained by %s for %s epochs' % ('ceb' if FLAGS.use_of_ceb else 'vib', FLAGS.num_epochs))
        if FLAGS.cifar10:
            X_test, y_test = build_test_set(test_loader, device, flatten=False)
        else:
            X_test, y_test = build_test_set(test_loader, device)

        dnn_input_units = X_test.shape[1] if FLAGS.cifar10 else X_test.shape[-1]
        del(X_test, y_test)
        
        dnn_output_units = FLAGS.num_classes
        if FLAGS.enc_type == 'CNN':
            print('Building CNN')
            Encoder = CNN(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device)
        elif FLAGS.enc_type =='VAE':
            print('Building VAE')
            Encoder = VAE(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device) 
        else:
            print('Building MLP')
            Encoder = MLP(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device)
        Encoder.load_state_dict(torch.load('pretrained_encoders/enc_%s_%sepochs.pt' % ('ceb' if FLAGS.use_of_ceb else 'vib', FLAGS.num_epochs)))
    else:
        if FLAGS.use_of_vib or FLAGS.use_of_ceb:
            Encoder = train_encoder_VIB(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device)
        else:
            Encoder = train_encoder(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device)
        if FLAGS.save_encoder:
            torch.save(Encoder.state_dict(), 'pretrained_encoders/enc_%s_%sepochs.pt' % ('ceb' if FLAGS.use_of_ceb else 'vib', FLAGS.num_epochs))
            print('Saved encoder that has been trained for %s epochs' % FLAGS.num_epochs)
    # '''
    '''
    if FLAGS.use_of_vib or FLAGS.use_of_ceb:
        Encoder = train_encoder_VIB(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device)
    else:
        Encoder = train_encoder(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device)
    # '''
    print('Best achieved performance: %s \n' % Encoder.best_performance)
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

    if FLAGS.num_seeds:
        seeds = np.arange(1, FLAGS.num_seeds + 1)
    print('Seeds to evaluate - ', seeds)

    if FLAGS.mnist12k:
            train_set_to_split = MNIST('./data/MNIST', download=True, train=True, transform=ToTensor())
    else:
            train_set_to_split = train_set
    
    path_to_subsets = 'label_partitions' + '/cifar10' if FLAGS.cifar10 else '/mnist'
    if FLAGS.load_subsets:
        if not os.path.exists(path_to_subsets):
            time_start = time.time()
            print('Performing custom split to get training subsets with different amount of labeled examples')
            train_subsets = build_training_subsets(train_set_to_split, base=2, num_classes=FLAGS.num_classes)
            os.makedirs(path_to_subsets)
            for key in train_subsets.keys():
                torch.save(train_subsets[key], path_to_subsets + '/%s.pt' % key)
            print('Done custom split. Time spent: %s \n' % (time.time() - time_start))
        else:
            print('Loading pre-saved label partitions of the dataset')
            keys = [int(''.join(c for c in file_name if c.isdigit())) for file_name in os.listdir(path_to_subsets)]
            train_subsets = {}
            for key in keys:
                train_subsets[key] = torch.load(path_to_subsets + '/%s.pt' % key)
    else:
        print('Performing custom split to get training subsets with different amount of labeled examples')
        train_subsets = build_training_subsets(train_set_to_split, base=2, num_classes=FLAGS.num_classes)
        if not os.path.exists(path_to_subsets):
            os.makedirs(path_to_subsets)
        if not os.listdir(path_to_subsets):
            print('Saving partitions')
            for key in train_subsets.keys():
                torch.save(train_subsets[key], path_to_subsets + '/%s.pt' % key)
        else:
            print('No overwirting of partitions')

    num_labels_range = list(train_subsets.keys()) 
    acc = {layer:{i:[] for i in num_labels_range} for layer in layers_names}
    start_time = time.time()

    for i in range(len(seeds)):

        print('\nRunning with seed %d out of %d' % (i+1, len(seeds)))
        torch.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        seed(seeds[i])

        for layer in layers_names:
            for num_labels in num_labels_range:
                print('\n\nEvaluating for %s (labels per class - %d)' % (layer, int(num_labels/FLAGS.num_classes)))
                if FLAGS.enc_type == 'MLP':
                    train_accuracy, test_accuracy = evaluate(encoder=Encoder.models[layer], enc_type=FLAGS.enc_type, train_on=train_subsets[num_labels], test_on=test_set, eval_num_samples = FLAGS.eval_num_samples, cuda=torch.cuda.is_available())
                else:
                    train_accuracy, test_accuracy = evaluate(encoder=Encoder, enc_type=FLAGS.enc_type, train_on=train_subsets[num_labels], test_on=test_set, eval_num_samples = FLAGS.eval_num_samples, cuda=torch.cuda.is_available())

                print('Train Accuracy: %f'% train_accuracy)
                print('Test Accuracy: %f'% test_accuracy)
                
                acc[layer][num_labels].append(test_accuracy)
        print('Elapsed time - ', time.time() - start_time)

    if not os.path.exists(FLAGS.result_path):
          os.makedirs(FLAGS.result_path)
    acc_df = pd.DataFrame.from_dict(acc)
    acc_df.to_csv(FLAGS.result_path+'/acc_track.csv', sep=' ')
    
    abc_metric_values = plot_acc_numlabels(FLAGS, acc_df, layers_names, Encoder.best_performance, num_labels_range)
    pd.DataFrame(abc_metric_values, index=[0, 1]).to_csv(FLAGS.result_path+'/abc_values.csv', sep=' ')

if __name__=='__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = get_option(parser)

    global_start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available()

    if FLAGS.use_of_vib or FLAGS.use_of_ceb:
        FLAGS.enc_type = 'VAE'

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
    elif not FLAGS.cifar10:
        print('Uploading Regular MNIST')
        train_set = MNIST('./data/MNIST', download=True, train=True, transform=ToTensor())
        test_set = MNIST('./data/MNIST', download=True, train=False, transform=ToTensor())

    # Single time define the testing set. Keep it fixed until the end
    # if not FLAGS.cifar10:
    #  Initialization of the data loader
    train_loader = DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1)
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