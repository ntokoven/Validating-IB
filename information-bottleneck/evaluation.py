import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import torchvision.transforms.functional as F

class EmbeddedDataset:
    BLOCK_SIZE = 256

    def __init__(self, base_dataset, encoder, enc_type, eval_num_samples, test=False, cuda=True):
        if cuda:
            encoder = encoder.cuda()
        self.test = test
        self.eval_num_samples = eval_num_samples
        self.means, self.target = self._embed(encoder, enc_type, base_dataset, cuda)

    def _embed(self, encoder, enc_type, dataset, cuda):
        encoder.eval()

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.BLOCK_SIZE,
            shuffle=False)

        ys = []
        reps = []
        with torch.no_grad():
            for x, y in data_loader:
                if x.shape[1] == 1:
                    x = x.flatten(start_dim=1)
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                embeddings = []
                if enc_type == 'VAE':
                    (mu, std), _, p_z_given_x = encoder(x)
                    if self.test or self.eval_num_samples == 0:
                        reps.append(mu.detach()) #to sample mean (reduced variance, less noise - more accurate estimates of quality)
                    else:
                        samples = encoder.reparametrize_n(mu, std, self.eval_num_samples)
                        if samples.dim() == 2: #meaning that we use only 1 sample
                            samples = samples.unsqueeze(0)
                        reps.append(samples.mean(0).detach())
                else: 
                    p_z_given_x = encoder(x)
                    reps.append(p_z_given_x.detach())
                
                ys.append(y)
            ys = torch.cat(ys, 0)
        encoder.train()
        return reps, ys

    def __getitem__(self, index):
        y = self.target[index]
        x = self.means[index // self.BLOCK_SIZE][index % self.BLOCK_SIZE]
        return x, y

    def __len__(self):
        return self.target.size(0)

def train_and_evaluate_linear_model(train_set, test_set, solver='saga', multi_class='multinomial', tol=.1, C=10):
    model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
    scaler = MinMaxScaler()

    x_train, y_train = build_matrix(train_set)
    x_test, y_test = build_matrix(test_set)

    
    # For some set-ups NaN values can occur after scaling
    x_train = np.nan_to_num(scaler.fit_transform(x_train))
    x_test = np.nan_to_num(scaler.transform(x_test))

    model.fit(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    train_accuracy = model.score(x_train, y_train)

    return train_accuracy, test_accuracy

def split(dataset, size, split_type):
    if split_type == 'Random':
        data_split, _ = torch.utils.data.random_split(dataset, [size, len(dataset) - size])
    elif split_type == 'Balanced':
        class_ids = {}
        for idx, (_, y) in enumerate(dataset):
            if y not in class_ids:
                class_ids[y] = []
            class_ids[y].append(idx)

        ids_per_class = size // len(class_ids)

        selected_ids = []

        for ids in class_ids.values():
            selected_ids += list(np.random.choice(ids, min(ids_per_class, len(ids)), replace=False))
        data_split = Subset(dataset, selected_ids)

    return data_split

def build_matrix(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    xs = []
    ys = []

    for x, y in data_loader:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs, 0)
    ys = torch.cat(ys, 0)

    if xs.is_cuda:
        xs = xs.cpu()
    if ys.is_cuda:
        ys = ys.cpu()

    return xs.data.numpy(), ys.data.numpy()

def evaluate(encoder, enc_type, train_on, test_on, eval_num_samples, cuda):
    embedded_train = EmbeddedDataset(train_on, encoder, enc_type, eval_num_samples, cuda=cuda)
    embedded_test = EmbeddedDataset(test_on, encoder, enc_type, eval_num_samples, test=True, cuda=cuda)
    return train_and_evaluate_linear_model(embedded_train, embedded_test)

# Get partition of dataset into overlapping training subsets of labeled examples (subset_i \in subset_j for i, j \in every i < j)
def build_training_subsets(train_set, base=2, num_classes=10):
    
    n_train_samples = len(train_set) 

    # Different partition as increasing powers of base up to the size of the train_set
    labels_per_class = [base**i for i in range(0, int(np.log(n_train_samples/num_classes) / np.log(base)))] 
    num_labels_range = num_classes * np.array(labels_per_class)
    
    train_subsets = {}
    train_subsets[num_labels_range[0]] = split(train_set, num_labels_range[0], 'Balanced')
    for num_labels in num_labels_range[1:]:
        train_subset = split(train_set, num_labels//2, 'Balanced')
        train_subset.indices = train_subsets[num_labels/2].indices + train_subset.indices
        train_subsets[num_labels] = train_subset
    
    return train_subsets

   
 
