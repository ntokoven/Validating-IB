import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import softplus

import math
import time

from data_utils import *
from evaluation import *
from helper import *
import cifar10_utils
from models import CNN, VAE, MLP, MIEstimator

# Training deterministic encoder
def train_encoder(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device):

    if FLAGS.enc_type == 'CNN':
        X_test, y_test = build_test_set(test_loader, device, flatten=False)
    else:
        X_test, y_test = build_test_set(test_loader, device)

    dnn_input_units = X_test.shape[1] if FLAGS.enc_type == 'CNN' else X_test.shape[-1]
    dnn_output_units = FLAGS.num_classes

    if FLAGS.weight_decay != 0:
        print('\nWeight decay to be applied: ', FLAGS.weight_decay)
    if FLAGS.p_dropout != 0:
        print('\nApplying dropout with rate %s' % FLAGS.p_dropout)
    if FLAGS.enc_type == 'CNN':
        print('Building CNN')
        model = CNN(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device)
    elif FLAGS.enc_type =='VAE':
        print('Building VAE')
        model = VAE(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device) 
    else:
        print('Building MLP')
        model = MLP(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay) #default lr 1e-3
    
    start_time = time.time()
    max_accuracy = 0


    for epoch in range(FLAGS.num_epochs):
        # training loop 
        model.train()

        for X_train, y_train in train_loader:
            if FLAGS.enc_type != 'CNN':
                X_train, y_train = X_train.flatten(start_dim=1).to(device), y_train.flatten(start_dim=0).to(device) # discard the channel dimension
            else:
                X_train, y_train = X_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            if FLAGS.enc_type =='VAE':
                (mu, std), out, z_train = model(X_train)
                loss = criterion(out, y_train)
            else:
                out = model(X_train)
                loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
            train_accuracy = accuracy(out, y_train)
            del(X_train)
            del(y_train)
        with torch.no_grad():
            # evaluation
            if epoch % FLAGS.eval_freq == 0 or epoch == FLAGS.num_epochs - 1:
                    model.eval()

                    print('\n'+'#'*30)
                    print('Training epoch - %d/%d' % (epoch+1, FLAGS.num_epochs))

                    if FLAGS.enc_type == 'VAE':
                        (_, _), out_test, _ = model(X_test)
                        test_loss = criterion(out_test, y_test)
                    else:
                        out_test = model(X_test)
                        test_loss = criterion(out_test, y_test)

                    
                    test_accuracy = accuracy(out_test, y_test)
                    if test_accuracy > max_accuracy:
                        max_accuracy = test_accuracy

                    print('Train: Accuracy - %0.3f, Loss - %0.3f' % (train_accuracy, loss))
                    print('Test: Accuracy - %0.3f, Loss - %0.3f' % (test_accuracy, test_loss))
                    print('Elapsed time: ', time.time() - start_time)
                    print('#'*30,'\n')
                    if test_accuracy == 1 and test_loss == 0:
                        break
                    
    model.best_performance = max_accuracy
    return model.eval()

def train_encoder_VIB(FLAGS, encoder_hidden_units, decoder_hidden_units, train_loader, test_loader, device, dnn_input_units=784, dnn_output_units=10):
    if FLAGS.cifar10:
        X_test, y_test = build_test_set(test_loader, device, flatten=False)
    else:
        X_test, y_test = build_test_set(test_loader, device)

    dnn_input_units = X_test.shape[1] if FLAGS.enc_type == 'CNN' else X_test.shape[-1]
    z_dim = encoder_hidden_units[-1]
    dnn_output_units = FLAGS.num_classes

    model = VAE(dnn_input_units, encoder_hidden_units, decoder_hidden_units, dnn_output_units, FLAGS).to(device)
    
    criterion = nn.CrossEntropyLoss()
    if FLAGS.use_of_ceb:
        q_z_given_y = nn.Sequential(nn.Linear(dnn_output_units, 2 * z_dim)).to(device)
        optimizer = optim.Adam([
                    {'params': model.parameters(), 'lr':FLAGS.learning_rate}, 
                    {'params': q_z_given_y.parameters(), 'lr':FLAGS.learning_rate}, 
                    ])
    else:
        optimizer = optim.Adam(model.parameters(),lr=FLAGS.learning_rate,betas=(0.5,0.999))

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.97)
    
    start_time = time.time()
    max_accuracy = 0
    beta = FLAGS.vib_beta
    
    print('I am here doing ceb-%s, vib-%s' % (FLAGS.use_of_ceb, FLAGS.use_of_vib))

    for epoch in range(FLAGS.num_epochs):
        model.train()
        for X_train, y_train in train_loader:
            # beta = scheduler(epoch)
            if FLAGS.cifar10:
                X_train, y_train = X_train.to(device), y_train.to(device)
            else:
                X_train, y_train = X_train.flatten(start_dim=1).to(device), y_train.flatten(start_dim=0).to(device)


            optimizer.zero_grad()
            (mu, std), out, z_train = model(X_train)
            # encoding = model.reparametrize_n(mu, std, 1) # Can average across more samles
            # out = model.decode(encoding)
            
            class_loss = criterion(out, y_train).div(math.log(2)) #make log of base 2 
            if FLAGS.use_of_vib:
                info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean().div(math.log(2)) # KL(p(Z|x), r(Z))
            elif FLAGS.use_of_ceb:
                y = onehot_encoding(y_train, device=device).float()
                statistics_y = q_z_given_y(y) 
                mu_y = statistics_y[:,:z_dim]
                std_y = softplus(statistics_y[:,z_dim:]-5,beta=1)

                eps = model.reparametrize_n(torch.zeros_like(mu), torch.ones_like(std), 1) # get random noise

                '''
                # Original CEB loss formulation
                
                if beta == 1:
                    beta += 1e-5
                info_loss = 0.5 / (1 - beta) * ((mu - mu_y) @ (mu - mu_y + 2 * eps).t()).sum(1).mean()  # notice the reparametrization of beta
                '''

                p = torch.distributions.Normal(mu, std)
                q = torch.distributions.Normal(mu_y, std_y)
                sample_x = std * eps + mu

                kl_loss = (p.log_prob(sample_x) - q.log_prob(sample_x)).sum(dim=1).mean().div(math.log(2))
                
                info_loss = kl_loss
            total_loss = class_loss + beta * info_loss # H(p,q) + beta * KL(p(Z|x), r(Z))
            izy_bound = math.log(10,2) - class_loss # upperbound on entropy - empirical cross-entropy
            izx_bound = info_loss

            total_loss.backward()
            optimizer.step()

            train_accuracy = accuracy(out, y_train)
            
            del(X_train)
            del(y_train)

        with torch.no_grad():
            if epoch % FLAGS.eval_freq == 0 or epoch == FLAGS.num_epochs - 1:
                model.eval()
                print('\n'+'#'*30)
                print('Training epoch - %d/%d' % (epoch+1, FLAGS.num_epochs))

                (mu, std), out_test, z_test = model(X_test)
                test_class_loss = criterion(out_test, y_test).div(math.log(2)) #make log of base 2
                if FLAGS.use_of_vib:
                    test_info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2)) # KL(p(Z|x), r(Z))
                elif FLAGS.use_of_ceb:
                    y = onehot_encoding(y_test, device=device).float()
                    statistics_y = q_z_given_y(y) 
                    mu_y = statistics_y[:,:z_dim]
                    std_y = softplus(statistics_y[:,z_dim:]-5,beta=1)
                    eps = model.reparametrize_n(torch.zeros_like(mu), std, 1)
                    p = torch.distributions.Normal(mu, std)
                    q = torch.distributions.Normal(mu_y, std_y)
                    sample_x = std * eps + mu
                    torch_kl_loss = torch.distributions.kl_divergence(p, q).sum(1).mean().div(math.log(2))
                    kl_loss = (p.log_prob(sample_x) - q.log_prob(sample_x)).sum(dim=1).mean().div(math.log(2))
                    test_info_loss = kl_loss #0.5 / (1 - beta) * ((mu - mu_y) @ (mu - mu_y + 2 * eps).t()).mean()  
                    print('kl(p,q) | E_p log p/q = %s | %s ' % (torch_kl_loss.item(), kl_loss.item()))
                    

                test_total_loss = test_class_loss + beta * test_info_loss
                test_accuracy = accuracy(out_test, y_test)
                if test_accuracy > max_accuracy:
                    max_accuracy = test_accuracy

                print('Train: Accuracy - %0.3f, Loss - %0.3f' % (train_accuracy, total_loss))
                print('Test: Accuracy - %0.3f, Loss - %0.3f' % (test_accuracy, test_total_loss))
                print('Upperbound I(X, T)', izx_bound.item())
                print('Lowerbound I(T, Y)', izy_bound.item())
                print('Beta = %s' % FLAGS.vib_beta)
                print('Elapsed time: ', time.time() - start_time)
                print('#'*30,'\n')
                if test_accuracy == 1 and test_total_loss == 0:
                    break
    model.best_performance = max_accuracy
    return model.eval()


