import torch
import torch.optim as optim
import torch.nn as nn

import math
import time

from data_utils import *
from evaluation import *
from helper import *
from models import VAE, MLP, MIEstimator


def train_encoder(FLAGS, dnn_hidden_units, train_loader, test_loader, device, dnn_input_units=784, dnn_output_units=10):
    if FLAGS.weight_decay != 0:
        print('\nWeight decay to be applied: ', FLAGS.weight_decay)
    if FLAGS.p_dropout != 0:
        print('Applying dropout with rate %s' % FLAGS.p_dropout)
    if FLAGS.enc_type == 'MLP':
        model = MLP(dnn_input_units, dnn_hidden_units, dnn_output_units, FLAGS).to(device)
    elif FLAGS.enc_type =='VAE':
        model = VAE(dnn_input_units, dnn_hidden_units, dnn_output_units, FLAGS).to(device) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay) #default lr 1e-3
    
    X_test, y_test = build_test_set(test_loader, device)

    start_time = time.time()
    max_accuracy = 0

    for epoch in range(FLAGS.num_epochs):
        # training loop 
        model.train()

        for X_train, y_train in train_loader:
            X_train, y_train = X_train.flatten(start_dim=1).to(device), y_train.to(device)
            optimizer.zero_grad()
            if FLAGS.enc_type =='VAE':
                (mu, std), out, z_train = model(X_train)
                loss = criterion(out, y_train)
            else:
                out = model(X_train)
                loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()

        # evaluation
        if epoch % FLAGS.eval_freq == 0 or epoch == FLAGS.num_epochs - 1:
                model.eval()

                print('\n'+'#'*30)
                print('Training epoch - %d/%d' % (epoch+1, FLAGS.num_epochs))

                if FLAGS.enc_type == 'VAE':
                    (_, _), out_train, _ = model(X_train)
                    train_loss = criterion(out_train, y_train)

                    (_, _), out_test, _ = model(X_test)
                    test_loss = criterion(out_test, y_test)
                else:
                    out_train = model(X_train)
                    train_loss = criterion(out_train, y_train)

                    out_test = model(X_test)
                    test_loss = criterion(out_test, y_test)

                train_accuracy = accuracy(out_train, y_train)
                test_accuracy = accuracy(out_test, y_test)
                if test_accuracy > max_accuracy:
                    max_accuracy = test_accuracy

                print('Train: Accuracy - %0.3f, Loss - %0.3f' % (train_accuracy, train_loss))
                print('Test: Accuracy - %0.3f, Loss - %0.3f' % (test_accuracy, test_loss))
                print('Elapsed time: ', time.time() - start_time)
                print('#'*30,'\n')
                if test_accuracy == 1 and test_loss == 0:
                    break

    model.best_performance = max_accuracy
    return model.eval()

def train_encoder_VIB(FLAGS, dnn_hidden_units, train_loader, test_loader, device, dnn_input_units=784, dnn_output_units=10):

    model = VAE(dnn_input_units, dnn_hidden_units, dnn_output_units, FLAGS).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-4,betas=(0.5,0.999))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.97)
    
    X_test, y_test = build_test_set(test_loader, device)
    
    start_time = time.time()
    max_accuracy = 0
    beta = FLAGS.vib_beta

    for epoch in range(FLAGS.num_epochs):
        for X_train, y_train in train_loader:
            # beta = scheduler(epoch)
            
            X_train, y_train = X_train.flatten(start_dim=1).to(device), y_train.to(device)
            optimizer.zero_grad()
            (mu, std), out, z_train = model(X_train)
            
            class_loss = criterion(out, y_train).div(math.log(2)) #make log of base 2
            info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
            total_loss = class_loss + beta * info_loss # -(H(p,q) - beta * I(X,z))

            izy_bound = math.log(10,2) - class_loss # upperbound on entropy - empirical cross-entropy
            izx_bound = info_loss

            total_loss.backward()
            optimizer.step()

        if epoch % FLAGS.eval_freq == 0 or epoch == FLAGS.num_epochs - 1:

                print('\n'+'#'*30)
                print('Training epoch - %d/%d' % (epoch+1, FLAGS.num_epochs))

                (mu, std), out_test, z_test = model(X_test)
                test_class_loss = criterion(out, y_train).div(math.log(2)) #make log of base 2
                test_info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
                test_total_loss = test_class_loss + beta * test_info_loss
                test_accuracy = accuracy(out_test, y_test)
                if test_accuracy > max_accuracy:
                    max_accuracy = test_accuracy

                print('Train: Accuracy - %0.3f, Loss - %0.3f' % (accuracy(out, y_train), total_loss))
                print('Test: Accuracy - %0.3f, Loss - %0.3f' % (test_accuracy, test_total_loss))
                print('Upperbound I(X, T)', izx_bound.item())
                print('Lowerbound I(T, Y)', izy_bound.item())
                print('Beta = %s' % FLAGS.vib_beta)
                print('Elapsed time: ', time.time() - start_time)
                print('#'*30,'\n')
                if test_accuracy == 1 and test_total_loss == 0:
                    break
    model.best_performance = max_accuracy
    return model