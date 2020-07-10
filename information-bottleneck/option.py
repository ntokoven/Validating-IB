def get_option(parser):
    parser.add_argument('--enc_type', type = str, default = 'MLP',
                        help='Type of encoder to train')
    parser.add_argument('--p_dropout', type = float, default = 0,
                        help='Probability of dropout')
    parser.add_argument('--p_input_dropout', type = float, default = 0,
                        help='Probability of dropout')
    parser.add_argument('--encoder_hidden_units', type = str, default = '1024,1024,256',
                        help='Comma separated list of number of units in each hidden layer of encoder')
    parser.add_argument('--decoder_hidden_units', type = str, default = '64,64',
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
    parser.add_argument('--clip_grad_norm', type = float, default = 1e-3,
                        help='Value of weights norm to clip')
    parser.add_argument('--vib_beta', type = float, default = 1e-3,
                        help='Lagrangian multiplier representing prioirity of MI(z, y) over MI(x, z)')
    parser.add_argument('--use_of_vib', type = bool, default = False,
                        help='Need to train using Variational Information Bottleneck objective')
    parser.add_argument('--use_of_ceb', type = bool, default = False,
                        help='Need to train using Conditional Entropy Bottleneck objective')
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
    parser.add_argument('--eval_freq', type=int, default=10,
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
    parser.add_argument('--cifar10', type = bool,
                      help='Run for CIFAR10')
    parser.add_argument('--load_subsets', type = bool,
                      help='If need to perform custom split to get training subsets with different amount of labeled examples')
    parser.add_argument('--use_scheduler', type = bool,
                      help='Use Exponential Scheduler for beta')
    parser.add_argument('--eval_num_samples', type = int, default = 1,
                      help='Number of samples to evaluate the encoder (if 0, then using mean of the posterior')
    parser.add_argument('--save_encoder', type = bool,
                      help='Need tp save trained encoder')

    # TODO: split arguments for MIE and GC separately
    
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed