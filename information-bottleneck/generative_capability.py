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

def plot_acc_numlabels(acc_df, layers_names, best_performance, num_labels_range):
    
    fig1, ax = plt.subplots(len(layers_names), 1, sharex=True)
    nums = np.array(num_labels_range)/num_classes
    colors = ['black', 'blue', 'red', 'green', 'yellow']

    mean = lambda x: np.mean(x)
    std = lambda x: np.std(x)

    for i in range(len(layers_names)):
        means = acc_df[layers_names[i]].apply(mean)
        stds = acc_df[layers_names[i]].apply(std)
        metric_value = abc_metric(best_performance, acc_df[layers_names[i]].apply(mean).to_numpy(), nums)
        if len(layers_names) != 1:
            tx = ax[i]
        else:
            tx = ax
        tx.plot(np.log10(nums), means, color=colors[i], label=layers_names[i]+' + weight_decay (%s) => ABC=%0.5f' % (weight_decay, metric_value))
        tx.plot(np.log10(nums), best_performance * np.ones(len(nums)))
        tx.fill_between(np.log10(nums), means - stds, means + stds, facecolor=colors[i], interpolate=True, alpha=0.25)
        tx.grid()
        tx.set_ylim((np.min(acc_df.min().min()), 1))

    tx.set_xlabel('# Labels per class (log10)')
    tx.set_ylabel('Accuracy')

    fig1.legend()
    fig1.set_size_inches(10, 7, forward=True)
    if not os.path.exists(FLAGS.result_path):
          os.makedirs(FLAGS.result_path)
    fig1.savefig(FLAGS.result_path+'/acc_%s_%s.png' % (enc_type.lower(), int(1/weight_decay) if weight_decay != 0 else 0))

def abc_metric(best_performance, accuracy_over_labels, nums):
    perf = best_performance * np.ones(len(nums))
    x_y_curve1 = [(np.log10(nums)[i], accuracy_over_labels[i]) for i in range(len(nums))]
    x_y_curve2 = [(np.log10(nums)[i], perf[i]) for i in range(len(nums))] 

    polygon_points = [] #creates a empty list where we will append the points to create the polygon

    for xyvalue in x_y_curve1:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1

    for xyvalue in x_y_curve2[::-1]:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

    for xyvalue in x_y_curve1[0:1]:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

    polygon = Polygon(polygon_points)
    area = polygon.area
    return area

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    
    Encoder = train_encoder(dnn_hidden_units, enc_type=enc_type, num_epochs=num_epochs, weight_decay=weight_decay)
    print('Best achieved performance: ', Encoder.best_performance)
    print(Encoder)

    print('Performing custom split to get training subsets with different amount of labeled examples')
    train_subsets = build_training_subsets(train_set, base=2, num_classes=num_classes)
    num_labels_range = list(train_subsets.keys()) 
    print('Done custom split \n')

    if FLAGS.layers_to_track:
        layers_to_track = FLAGS.layers_to_track.split(",")
        layers_to_track = [int(layer_num) for layer_num in layers_to_track]
    else:
        layers_to_track = [1]

    pos_layers = - (np.array(layers_to_track) + 1)
    layers_names = []
    for pos in pos_layers:
        layers_names.append(list(get_named_layers(Encoder).keys())[pos].split('_')[0])

    acc = {layer:{i:[] for i in num_labels_range} for layer in layers_names}

    if FLAGS.seeds:
        seeds = FLAGS.seeds.split(",")
        seeds = [int(seed) for seed in seeds]
    else:
        seeds = [default_seed]

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
    plot_acc_numlabels(acc_df, layers_names, Encoder.best_performance, num_labels_range)





if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--enc_type', type = str, default = 'MLP',
                        help='Type of encoder to train')
    parser.add_argument('--dnn_hidden_units', type = str, default = '1024,512,256,128,64',
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--default_seed', type = int, default = 69,
                        help='Default seed for encoder training')
    parser.add_argument('--seeds', type = str, default = '9,42,103,48,79',
                        help='Comma separated list of random seeds')
    parser.add_argument('--layers_to_track', type = str, default = '1',
                        help='Comma separated list of inverse positions of encoding layers to evaluate (starting from 1)')
    parser.add_argument('--learning_rate', type = float, default = 1e-3,
                        help='Learning rate for encoder training')
    parser.add_argument('--weight_decay', type = float, default = 0,
                      help='Value of weight decay applied to optimizer')
    parser.add_argument('--num_epochs', type = int, default = 10,
                        help='Number of epochs to do training')
    parser.add_argument('--num_classes', type = int, default = 10,
                        help='Number of classes')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=1,
                            help='Frequency of evaluation on the test set')
    parser.add_argument('--neg_slope', type=float, default=0.02,
                        help='Negative slope parameter for LeakyReLU')
    parser.add_argument('--result_path', type = str, default = 'results',
                      help='Directory for storing results')

    FLAGS, unparsed = parser.parse_known_args()

    print_flags()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available()

    default_seed = FLAGS.default_seed 
    num_classes = FLAGS.num_classes
    batch_size = FLAGS.batch_size 
    enc_type = FLAGS.enc_type
    weight_decay = FLAGS.weight_decay 
    num_epochs = FLAGS.num_epochs
    learning_rate = FLAGS.learning_rate

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