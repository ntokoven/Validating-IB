import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from collections import Counter, OrderedDict
from shapely.geometry import Polygon

# Map integer labels to onehot encoding
def onehot_encoding(x, num_classes=10, device=torch.device('cpu')):
    x_onehot = torch.FloatTensor(x.shape[0], num_classes).to(device)
    x_onehot.zero_()
    x_onehot.scatter_(1, x.view(x.shape[0], 1), 1)
    return x_onehot

def whitening(x, batch_dim=0):
    # Normalize distribution of the input with respect to the batch dimension
    mean = x.mean(dim=batch_dim)
    std = x.std(dim=batch_dim)
    x = (x - mean) / std
    # NaN values might occur if certain dimension always takes 0 value for all examples in the batch, so std is 0 as well
    x[x != x] = 0 
    x[x == -float('inf')] = 0
    x[x == float('inf')] = 0
    return x

def concat(names_list):
    names_as_strings = ['%s_' % l for l in names_list[:-1]] + ['%s' % names_list[-1]]
    s = ''
    for name in names_as_strings:
        s += name
    return s

# Determine the architecture of encoder
def get_named_layers(net):
    conv2d_idx = 0
    convT2d_idx = 0
    linear_idx = 0
    batchnorm2d_idx = 0
    named_layers = OrderedDict()
    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            layer_name = 'Conv2d{}_{}-{}'.format(
                conv2d_idx, mod.in_channels, mod.out_channels
            )
            named_layers[layer_name] = mod
            conv2d_idx += 1
        elif isinstance(mod, torch.nn.ConvTranspose2d):
            layer_name = 'ConvT2d{}_{}-{}'.format(
                conv2d_idx, mod.in_channels, mod.out_channels
            )
            named_layers[layer_name] = mod
            convT2d_idx += 1
        elif isinstance(mod, torch.nn.BatchNorm2d):
            layer_name = 'BatchNorm2D{}_{}'.format(
                batchnorm2d_idx, mod.num_features)
            named_layers[layer_name] = mod
            batchnorm2d_idx += 1
        elif isinstance(mod, torch.nn.Linear):
            layer_name = 'Linear{}_{}-{}'.format(
                linear_idx, mod.in_features, mod.out_features
            )
            named_layers[layer_name] = mod
            linear_idx += 1
    return named_layers

def accuracy(predictions, targets):
    """
    Compute accuracy given prediction logit
    """
    if targets.ndimension() == 1:
        accuracy = (predictions.argmax(dim=1) == targets).type(torch.FloatTensor).mean().item()
    else:
        accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).type(torch.FloatTensor).mean().item()
    return accuracy

def build_test_set(test_loader, device):
    """
    Build training set given test data loader
    """
    xs = []
    ys = []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs, 0)
    ys = torch.cat(ys, 0)
    
    X_test, y_test = xs.clone().detach().flatten(start_dim=1).to(device), ys.clone().detach().type(torch.LongTensor).flatten(start_dim=0).to(device)
    return X_test, y_test


def print_flags(flags):
    """
    Prints all entries in FLAGS variable.
    """
    print('\n'+'-'*30)
    for key, value in vars(flags).items():
        print(key + ' : ' + str(value))
    print('-'*30, '\n')


def rand_color():
    r = np.random.randint(0, 255)
    return '#%02X%02X%02X' % (r(),r(),r())

def get_colors(num):
    def rand_color():
        r = np.random.randint(0, 255)
        return '#%02X%02X%02X' % (r(),r(),r())

    if num < 8:
        colors = ['black', 'blue', 'red', 'green', 'yellow', 'cyan', 'magenta']
    else:
        colors = [rand_color() for _ in range(num)]
    return colors

def get_info(FLAGS):
    info = ''
    if FLAGS.use_of_vib:
        info += 'VIB beta = %s' % FLAGS.vib_beta
    elif FLAGS.enc_type == 'VAE':
        info += 'VAE'
    else:
        info += 'MLP'
    if FLAGS.p_dropout != 0:
        info += ' dout = %s' % FLAGS.p_dropout
    if FLAGS.weight_decay != 0:
        info += ' wd = %s' % FLAGS.weight_decay
    return info

def abc_metric(best_performance, accuracy_over_labels, nums, top=True):
    perf = best_performance * np.ones(len(nums))
    x_y_curve1 = [(np.log10(nums)[i], accuracy_over_labels[i]) for i in range(len(nums))]
    if top:
        x_y_curve2 = [(np.log10(nums)[i], perf[i]) for i in range(len(nums))] 
    else:
        x_y_curve2 = [(np.log10(nums)[i], 1) for i in range(len(nums))] 
    polygon_points = [] #creates an empty list where we will append the points to create the polygon

    for xyvalue in x_y_curve1:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1

    for xyvalue in x_y_curve2[::-1]:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

    for xyvalue in x_y_curve1[0:1]:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

    polygon = Polygon(polygon_points)
    area = polygon.area
    return area

def plot_acc_numlabels(FLAGS, acc_df, layers_names, best_performance, num_labels_range):
    
    fig1, ax = plt.subplots(len(layers_names), 1, sharex=True)
    nums = np.array(num_labels_range)/FLAGS.num_classes
    colors = get_colors(len(layers_names))

    mean = lambda x: np.mean(x)
    std = lambda x: np.std(x)
    abc_metric_values = {}
    for i in range(len(layers_names)):
        means = acc_df[layers_names[i]].apply(mean)
        stds = acc_df[layers_names[i]].apply(std)
        abc_metric_values[layers_names[i]] = metric_value = abc_metric(best_performance, acc_df[layers_names[i]].apply(mean).to_numpy(), nums), abc_metric(best_performance, acc_df[layers_names[i]].apply(mean).to_numpy(), nums, top=False)
        if len(layers_names) != 1:
            tx = ax[i]
        else:
            tx = ax
        tx.plot(np.log10(nums), means, color=colors[i], label=layers_names[i]+' + %s => ABC=(%0.5f, %0.5f)' % (get_info(FLAGS), metric_value[0], metric_value[1]))
        tx.plot(np.log10(nums), best_performance * np.ones(len(nums)))
        tx.fill_between(np.log10(nums), means - stds, means + stds, facecolor=colors[i], interpolate=True, alpha=0.25)
        tx.grid()
        # tx.set_ylim((np.min(acc_df.min().min()), 1))
        tx.set_ylim((0, 1))

    tx.set_xlabel('# Labels per class (log10)')
    tx.set_ylabel('Accuracy')

    fig1.legend()
    fig1.set_size_inches(10, 7, forward=True)
    if not os.path.exists(FLAGS.result_path):
          os.makedirs(FLAGS.result_path)
    fig1.savefig(FLAGS.result_path+'/acc_numlabels.png')
    plt.close(fig1)

    return abc_metric_values



def plot_mie_curve(FLAGS, mi_df, layer, seed):
    if not os.path.exists(FLAGS.result_path+'/mie_curves'):
        os.makedirs(FLAGS.result_path+'/mie_curves')
    if layer != '':
        layer_info = ' - layer %s, enc %s' % (layer, FLAGS.enc_type)
    else:
        layer_info = ''
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(np.arange(len(mi_df)), mi_df['X'], label='I(X,Z)%s' % layer_info)
    ax.plot(np.arange(len(mi_df)), mi_df['Y'], label='I(Z,Y)%s' % layer_info)
    
    ax.set_xlabel('# of training steps')
    ax.set_ylabel('MI Estimation')

    fig.legend()
    fig.set_size_inches(10, 7, forward=True)
    fig.savefig(FLAGS.result_path+'/mie_curves/mie_curve_%s_%s_l%s_s%s.png' % (FLAGS.enc_type.lower(), 'test' if FLAGS.mie_on_test else 'train', layer, seed))
    plt.close(fig)

def build_information_plane(mi_values, layers_names, seeds, FLAGS):
    mi_df = pd.DataFrame.from_dict(mi_values)
    fig2, ax2 = plt.subplots(1, 1, sharex=True)
    set_legend = True
    colors = get_colors(len(layers_names))
    for i in range(len(seeds)):
        for j in range(len(layers_names)):
            layer_num = int(re.sub(r"\D", "", layers_names[j]))
            if set_legend:
                ax2.scatter(mi_df.loc[seeds[i], layers_names[j]][0], mi_df.loc[seeds[i], layers_names[j]][1], color=colors[layer_num], label=layers_names[j])
            else:
                ax2.scatter(mi_df.loc[seeds[i], layers_names[j]][0], mi_df.loc[seeds[i], layers_names[j]][1], color=colors[layer_num])
            ax2.annotate('  seed %d' % seeds[i], (mi_df.loc[seeds[i], layers_names[j]][0], mi_df.loc[seeds[i], layers_names[j]][1]))
            ax2.grid()
        set_legend = False

    ax2.set_xlabel('I(X, Z)')
    ax2.set_ylabel('I(Z, Y)')

    fig2.legend()
    fig2.set_size_inches(10, 7, forward=True)
    if not os.path.exists(FLAGS.result_path+'/information_planes'):
        os.makedirs(FLAGS.result_path+'/information_planes')
    fig2.savefig(FLAGS.result_path+'/information_planes/info_plane_%s_%s.png' % (FLAGS.enc_type.lower(), 'test' if FLAGS.mie_on_test else 'train'))
    plt.close(fig2)


###############################################
# Helper functions to work with the Toy dataset
###############################################

#Calculate values MI for known distributions (for evaluating Toy example)
def calc_mutual_information(hidden, x_train_int, y_train):
    n_train_samples = len(x_train_int)
    n_neurons = hidden.shape[-1]
  
    # discretization 
    n_bins = 30
    bins = np.linspace(-1, 1, n_bins+1)
    indices = np.digitize(hidden, bins)
    
    # initialize pdfs
    pdf_x = Counter(); pdf_y = Counter(); pdf_t = Counter(); pdf_xt = Counter(); pdf_yt = Counter()

    for i in range(n_train_samples):
        pdf_x[x_train_int[i]] += 1/float(n_train_samples)
        pdf_y[y_train[i,0]] += 1/float(n_train_samples)      
        pdf_xt[(x_train_int[i],)+tuple(indices[i,:])] += 1/float(n_train_samples)
        pdf_yt[(y_train[i,0],)+tuple(indices[i,:])] += 1/float(n_train_samples)
        pdf_t[tuple(indices[i,:])] += 1/float(n_train_samples)
    
    # calcuate encoder mutual information I(X;T)
    mi_xt = 0
    for i in pdf_xt:
        # P(x,t), P(x) and P(t)
        p_xt = pdf_xt[i]; p_x = pdf_x[i[0]]; p_t = pdf_t[i[1:]]
        # I(X;T)
        mi_xt += p_xt * np.log(p_xt / p_x / p_t)
 
    # calculate decoder mutual information I(T;Y)
    mi_ty = 0
    for i in pdf_yt:
        # P(t,y), P(t) and P(y)
        p_yt = pdf_yt[i]; p_t = pdf_t[i[1:]]; p_y = pdf_y[i[0]]
        # I(T;Y)
        try:
          mi_ty += p_yt * np.log(p_yt / p_t / p_y)
        except ZeroDivisionError:
          mi_ty += p_yt * np.log(p_yt / (p_t + 1e-5) / (p_y + 1e-5))
            
    return mi_xt, mi_ty

# get mutual information for all hidden layers
def get_mutual_information(hidden):
    mi_xt_list = []; mi_ty_list = []
    # for hidden in hiddens:
    if True:
        mi_xt, mi_ty = calc_mutual_information(hidden)
        mi_xt_list.append(mi_xt)
        mi_ty_list.append(mi_ty)
    return mi_xt_list, mi_ty_list