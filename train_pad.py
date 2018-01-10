from __future__ import print_function
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.autograd import Variable
import torch.multiprocessing as mp
import pandas as pd
import os.path

import pipeline.feature_selection_wrapper as feat_sel

from torch.utils.data import Dataset, DataLoader, TensorDataset
import sklearn.model_selection
from sklearn.metrics import f1_score, confusion_matrix
import sklearn.metrics
from pipeline.hadmid_reader import Hadm_Id_Reader
import learning.deepLearn.model.rnn_pad as rnn_pad
import learning.deepLearn.model.attention_rnn_pad as attention_rnn_pad
import learning.deepLearn.model.cnn_rnn_pad as cnn_rnn_pad
import learning.deepLearn.model.attention_cnn_rnn_pad as attention_cnn_rnn_pad
from sklearn.model_selection import StratifiedShuffleSplit
import itertools
import time

startTime = time.time()

# Training settings
parser = argparse.ArgumentParser(description='Physionet Challenge 2017')
# Hyperparameters
parser.add_argument('--model', type=str, metavar='M', default='AttnCNNLSTM',
                    choices=['LSTM', 'AttnLSTM','CNNLSTM','AttnCNNLSTM'],
                    help='which model to train/evaluate')
parser.add_argument('--lr', type=float, metavar='LR', default=0.5,
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--momentum', type=float, metavar='M', default=0.5,
                    help='SGD momentum')
parser.add_argument('--batch-size', type=int, metavar='N', default=1000,
                    help='input batch size for training')
parser.add_argument('--num-rnn-layers',type=int, default=1,
                    help='number of rnn layers')
parser.add_argument('--epochs', type=int, metavar='N', default=10000,
                    help='number of epochs to train')
parser.add_argument('--hidden-dim', type=int, default=50,
                    help='number of hidden features/activations')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--totaltime', type=int, default=24, help='Total time to consider for reader. Must be divisible by 6')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='number of batches between logging train status')
parser.add_argument('--do-early-stopping',type = bool, default=False,
                    help='Do early stopping?')
parser.add_argument('--patience', type=int, default=50,
                    help='If early stopping is set, how many times will you wait?')
parser.add_argument('--folder', type=str, default='../saved_model/',
                    help='The output folder to save a model')
parser.add_argument('--model-num', type=str, default='test',
                    help='The naming number of the model to be saved')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class MIMIC3Dataset(Dataset):
    def __init__(self, hadm_dir, label_file, hadm_file_name="episode_timeseries.csv", idx=None, n_jobs_reading=20, transform=None):
        """
        This is a class for directly reading in the data from a directory structures where the events have been transformed
            into timeseries data and segmented into folders with the hospital admission id (hadm_id) as the name of the folders


        :param hadm_dir the directory where the hospital admission files are stored
        :param label_file the location where the labels (angus) for hospital admissions are located
        :param hadm_file_name the name of each admission file
        :param n_jobs_reading represents the number of jobs to use to read and reformat in the data
        :param idx a list of actual hospital admissions to use (by index, not actual hadm_id). If set to None, not considered
        :param transform an optional transforming function
        """
        self.reader = Hadm_Id_Reader(hadm_dir, file_name=hadm_file_name)
        self.reader.use_multiprocessing(n_jobs_reading)
        if idx is not None:
            self.reader.hadms = [self.reader.hadms[i] for i in idx]
        self.label = pd.DataFrame.from_csv(label_file)["angus"].loc[ \
                                [int(hadm) for hadm in self.reader.hadms]]
        print("beginning feature selection on training + validation set")
        X = self.reader.getFullAvg(endbound=args.totaltime)
        train_plus_idx, test_idx = sklearn.model_selection.train_test_split(X.index,\
                                                    test_size=0.1, stratify = self.label, random_state=args.seed)
        toKeep = feat_sel.chi2test(X.loc[train_plus_idx], self.label.loc[train_plus_idx], pval_thresh=.05)
        self.reader.set_features(toKeep.index)
        print("beginning reading")
        self.hadmToData = self.reader.get_time_matrices(total_time=args.totaltime)
        self.transform = transform
        self.num_features = len(toKeep.index)
    def __len__(self):
        return len(self.reader.hadms)


    def __getitem__(self, idx):
        hadms_to_return = [int(self.reader.hadms[i]) for i in idx]
        labels = self.label.loc[hadms_to_return]
        #turn from dictionary format into correct order in ndarray type
        # data = np.zeros((5, len(list(idx)), len(self.reader.__vars_to_keep)))
        # for i in range(5):
        #     for j in range(len(list(idx))):
        #         data[i, j, :] = self.hadmToData[int(self.reader.hadms[idx[j]])].iloc[i].as_matrix()
        data = np.array([self.hadmToData[hadm].as_matrix() for hadm in hadms_to_return])
        sample = {'data': data, 'label': labels.as_matrix(), 'lengths': np.array([datum.shape[0] for datum in data])}
        if self.transform is not None:
            return self.transform(sample)
        return sample


class PhysionetChallengeDataset(Dataset):
    """Physionet Challenge 2017 dataset."""

    def __init__(self, label_file=None, data_file=None,length_file=None, label=None,data=None,lengths=None, normalize=False, isTensor= False, transform=None):
        """
        Args:
            label_file (string): Path to the pickle file of labels.
            data_file (string): Path to the pickle file of data.
            length_file (string): Path to the pickle file of valid lengths of data
            normalize (callable, optional): Optional normalization (zero-center) to be applied
                on a sample.
        """
        if (label_file is not None) and (data_file is not None):
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
            with open(label_file, 'rb') as f:
                self.label = np.stack(pickle.load(f))
        if (label is not None) and (data is not None) :
            self.data = data
            self.label = label
        if lengths is not None:
            self.lengths = lengths
        if length_file is not None:
            self.lengths = pickle.load(open(length_file,'rb'))
        self.normalize = normalize
        self.isTensor = isTensor
        self.transform = transform



    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data, label, lengths = self.data[idx], self.label[idx], self.lengths[idx]
        sample = {'data': data, 'label': label, 'lengths': lengths}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def truncate_and_reshape(self, keep_length, timesteps):
        if keep_length % timesteps == 0:
            self.data = self.data[:, 0: keep_length].reshape(-1, timesteps, int(keep_length/timesteps))
            new_lengths = []
            num_features = keep_length/ timesteps
            for i in range(len(self.lengths)):
                if self.lengths[i]> keep_length:
                    self.lengths[i] = keep_length
                new_lengths.append(int(self.lengths[i] / num_features))
            self.lengths = np.asarray(new_lengths)
        else:
            raise ValueError('Cannot reshape a tensor to keep its dimensions, need to set parameters correct!\n')



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label, lengths = sample['data'], sample['label'],sample['lengths']

        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(np.array(label)),
                'lengths': torch.from_numpy(np.array(lengths))}

    # Truncate to keep the keep_length of every sequence (self. data in 2D) and then reshape it to 3D with predefined
    # timesteps, update valid lengths

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = ['AF','Normal','Other','Noisy']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    thresh = 1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def train(epoch):
    '''
    Train the model for one epoch.
    '''
    # Some models use slightly different forward passes and train and test
    # time (e.g., any model with Dropout). This puts the model in train mode
    # (as opposed to eval mode) so it knows which one to use.
    model.train()
    # train loop
    runningLoss = 0;
    for batch_idx, batch in enumerate(trainloader):
        # Sort the input X in descending order in terms of the valid length of timesteps
        # sorted, indices = torch.sort(batch[], 0, descending=True)

        # prepare data
        # X, y, l = torch.index_select(batch['data'],0,torch.squeeze(indices)),torch.index_select(batch['label'],0,torch.squeeze(indices)),\
        #           torch.index_select(batch['lengths'],0,torch.squeeze(indices))
        X = batch[0]
        y = batch[1]
        l = [X.size(1) for i in range(X.size(0))]
        #batch first is a thing... nvm....

        # newX = np.zeros((X.size(1), X.size(0), X.size(2))) #apparently its time by instance by features
        # for i in range(X.size(1)):
        #     for j in range(X.size(0)):
        #         # print(newX[i,j,:])
        #         # print(X[j, i, :])
        #         # print(X.size())
        #         newX[i,j,:] = X[j, i, :].numpy()
        # newY = torch.from_numpy(np.array([label for label in y for i in range(X.size(1))]))
        input, targets = Variable(X.float()), Variable(y)
        # l=torch.squeeze(l)
        if args.cuda:
            input, targets = input.cuda(), targets.cuda()

        # Init states
        #h, c = model.init_hidden(len(input))

        # Update the parameters in model using the optimizer from above.
        optimizer.zero_grad()
        output = model(input.float(), l)
        loss = criterion(output, torch.squeeze(targets))
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            val_loss, val_acc, val_f1 = evaluate(model, 'val', n_batches=4)
            train_loss = loss.data[0]
            runningLoss += train_loss
            examples_this_epoch = batch_idx * len(input)
            epoch_progress = 100. * batch_idx / len(trainloader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}\tVal F1:{:.6f}'.format(
                epoch, examples_this_epoch, len(trainloader.dataset),
                epoch_progress, train_loss, val_loss, val_acc, val_f1))

    return {'val_f1':val_f1, 'train_loss':runningLoss, "val_loss": val_loss}

def evaluate(model, split, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    old_dropout_layer = model.dropout_layer
    if split == 'val':
        loader = validloader
        model.dropout_layer = nn.Dropout(p=0.0) #Don't do dropout while validation or testing
    if split == 'test':
        loader = testloader
        model.dropout_layer = nn.Dropout(p=0.0) #Don't do dropout while validation or testing
    if split == 'train':
        loader = trainloader
    final_target = []
    final_pred = []
    for batch_i, batch in enumerate(loader):
        # Sort the input X in descending order in terms of the valid length of timesteps
        # sorted, indices = torch.sort(batch['lengths'], 0, descending=True)

        # prepare data
        # X, y, l = torch.index_select(batch['data'], 0, torch.squeeze(indices)), torch.index_select(batch['label'], 0,
                                                                    # torch.squeeze( indices)), torch.index_select(batch['lengths'], 0, torch.squeeze(indices))
        X, y = batch[0], batch[1],
        l = torch.FloatTensor([X.size(1) for i in range(X.size(0))])
        data, target = Variable(X, volatile= True), Variable(y)
        l = torch.squeeze(l)

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data, l.numpy())
        loss += criterion(output, torch.squeeze(target)).data[0]

        # predict the argmax of the log-probabilities
        pred = output.data.max(1)[1]
        ##output = model(data, list(l.numpy()), h, c)
        if args.cuda:
            final_target.append(target.view(len(data),).cpu().data.numpy().tolist())
            final_pred.append(pred.view(len(data), ).cpu().numpy().tolist())
        else:
            final_target.append(target.view(len(data), ).data.numpy().tolist())
            final_pred.append(pred.view(len(data),).numpy().tolist())
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += pred.size(0)
        if n_batches and (batch_i >= n_batches):
            break
    model.dropout_layer = old_dropout_layer
    loss /= n_examples
    acc = 100. * correct / n_examples
    final_pred = [item for sublist in final_pred for item in sublist]
    final_target = [item for sublist in final_target for item in sublist]
    f1 = f1_score(final_target, final_pred) #, average='macro')
    auc = sklearn.metrics.roc_auc_score(final_target, final_pred)
    mcc = sklearn.metrics.matthews_corrcoef(final_target, final_pred)
    prec = sklearn.metrics.precision_score(final_target, final_pred)
    recall = sklearn.metrics.recall_score(final_target, final_pred)
    # Neglect noisy class
    # f1 = np.mean(f1_score(final_target, final_pred, average=None)[0:3]) # not really a thing with sepsis data.
    '''
cnf_matrix = confusion_matrix(final_target, final_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=None, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
    '''
    if verbose:
        print('\nModel: {}, {} set, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1: {:.6f}, AUC: {:.6f}, MCC: {:.6f}), Precision: {:.6f}, Recall: {:.6f}\n'.format(
            args.model_num, split, loss, correct, n_examples, acc, f1, auc, mcc, prec, recall))
    return loss, acc, f1


def adjust_learning_rate(optimizer, lr):
    """Sets new learning rate of the optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_by_epoch(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 50 epochs"""
    lr = args.lr * (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# TODO: Ensembles of models trained on different sets of train/val set


# create training and testing vars and split train and test sets
# raw_data = pickle.load(open('../data/raw_data.pkl','rb'))
# length_list = [len(a) for a in raw_data]
# order_index = sorted(range(len(length_list)), key=lambda k: length_list[k], reverse =True)
# new_lengths = [length_list[i] for i in order_index]
# challenge_dataset = PhysionetChallengeDataset(label_file='../data/new_labels.pkl',
                                              # data_file='../data/new_padded_data.pkl', lengths=np.asarray(new_lengths), normalize=True)
challenge_dataset = MIMIC3Dataset(hadm_dir="data/rawdatafiles/byHadmID0", label_file="data/rawdatafiles/classifiedAngusSepsis.csv", transform=transforms.Compose([ToTensor()]))
# Keep the first 9000 values and reshape it to (N, 30, 3000)
# challenge_dataset.truncate_and_reshape(9000, 30)

# Binary classification
#challenge_dataset.label[challenge_dataset.label != 1] = 0



# Input size
# TODO: fix hardcoded portion
seq_size =(args.batch_size, args.totaltime / 6, challenge_dataset.num_features)

# Loss
criterion = nn.CrossEntropyLoss()


# Define model
if args.model == "LSTM":
    model = rnn_pad.LSTMModel(input_size=seq_size, n_classes=2, hidden_dim=args.hidden_dim,
                                    num_layers=args.num_rnn_layers, bidirection=0, flag_cuda=args.cuda)
elif args.model == "AttnLSTM":
    model = attention_rnn_pad.AttentionRecurrentModel(input_size=seq_size, n_classes=2, hidden_dim=args.hidden_dim,
                                                            flag_cuda=args.cuda, num_layers=args.num_rnn_layers)
elif args.model == 'CNNLSTM':
    model = cnn_rnn_pad.RecurrentConvModel(input_size = seq_size, n_classes=2, rnn_hidden_dim=args.hidden_dim,
                                                 cnn_kernel_size=9,cnn_output_channels=32, flag_cuda=args.cuda)
elif args.model == 'AttnCNNLSTM':
    model =  attention_cnn_rnn_pad.AttentionRecurrentConvModel(input_size = seq_size, n_classes=2, rnn_hidden_dim=args.hidden_dim,
                                                 cnn_kernel_size=9,cnn_output_channels=32, flag_cuda=args.cuda)

if args.cuda:
    model.cuda()


# Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Create TensorData and Dataloader
train_plus_idx, test_idx = sklearn.model_selection.train_test_split(list(range(len(challenge_dataset))), test_size=0.1, stratify = challenge_dataset.label, random_state=args.seed)
train_plus_set, test_set = challenge_dataset[train_plus_idx], challenge_dataset[test_idx]
test_tensor = TensorDataset(test_set['data'], test_set['label'])
testloader = DataLoader(test_tensor, batch_size=args.batch_size, shuffle=False)

#TEST
# best_model = torch.load('../saved_model/AttnLSTM_1.pt')
# evaluate(best_model, split='test', verbose=True)

# Stratified shuffle split to multiple sets of train - validation, set to one since we can just run multiple times
# Require numpy 0.18.0+ for this step
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

# watch the change of learning rate, for early stopping
lr_watch = args.lr

for train_index, valid_index in sss.split(train_plus_set['data'], list(train_plus_set['label'])):
    train_index = torch.from_numpy(train_index)
    valid_index = torch.from_numpy(valid_index)
    train_tensor = TensorDataset(train_plus_set['data'][train_index], train_plus_set['label'][train_index])

    '''
    # See whether model can overfit a small set of data
    train_tensor = PhysionetChallengeDataset(data=train_plus_set['data'][train_index[0:100]],
                                             label=train_plus_set['label'][train_index[0:100]],
                                             lengths=train_plus_set['lengths'][train_index[0:100]],
                                             transform=transforms.Compose([ToTensor()]))
   '''
    valid_tensor = TensorDataset(train_plus_set['data'][valid_index], train_plus_set['label'][valid_index])
    trainloader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(valid_tensor, batch_size=args.batch_size, shuffle=False)

    # train the model one epoch at a time
    max_metric = 0
    last_improved = 0

    # learning rate decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    results_to_write = pd.DataFrame(index=list(range(1, args.epochs + 1)), columns=['val_f1', 'train_loss', "val_loss"])
    for epoch in range(1, args.epochs + 1):
        result = train(epoch)
        val_score = result["val_f1"]
        results_to_write.loc[epoch, 'val_f1'] = result["val_f1"]
        results_to_write.loc[epoch, 'train_loss'] = result["train_loss"]
        results_to_write.loc[epoch, 'val_loss'] = result["val_loss"]
        if val_score > max_metric:
            last_improved = epoch
            max_metric = val_score
            # Save the model with largest val_score, currently set to F1 score, hardcoded
            torch.save(model, str(args.folder) + str(args.model) + '_' + str(args.model_num) + '.pt')
        if args.do_early_stopping is True:
            if (epoch - last_improved) > args.patience:
                lr_watch /= 2
                last_improved = epoch
                if lr_watch > 0.00000001:
                    adjust_learning_rate(optimizer, lr_watch)
                else:
                    break
        else:
            scheduler.step()
    evaluate(model, split='val', verbose=True)
    results_to_write.to_csv(os.path.join("data/rawdatafiles", "lstm", str(args.model_num) + ".csv"))
best_model = torch.load(str(args.folder) + str(args.model) +'_' + str(args.model_num)  + '.pt')
evaluate(best_model, split='test', verbose=True)

endTime = time.time()
print("Model:", args.model_num, ", Total time elapsed (in hours)", (endTime - startTime) / 3600)
