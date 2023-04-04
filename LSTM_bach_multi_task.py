import numpy as np
import math
import sys
import time
from collections import OrderedDict

from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler

from tensorboard.plugins.hparams import api as hp

from tqdm import tqdm

# gpu if available (global variable for convenience)
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

# to find float index in unique float list of standard scaled array
# works also for ints when not scaled
def uniqueLocation(uniques, note):
    for index, unique in enumerate(uniques):
        if (math.isclose(unique, note, abs_tol=0.0001)):
            return index
    return None

# returns concatenated onehot encoding for each note 
def one_hot_encode(y: np.ndarray, voices: np.ndarray) -> np.ndarray:
    # unique set of notes in the voice
    unique_voice1 = np.unique(voices[:,0])
    unique_voice2 = np.unique(voices[:,1])
    unique_voice3 = np.unique(voices[:,2])
    unique_voice4 = np.unique(voices[:,3])

    # initialize return arrays
    one_hot_voice1 = np.zeros((y.shape[0], len(unique_voice1)), dtype=np.float32)
    one_hot_voice2 = np.zeros((y.shape[0], len(unique_voice2)), dtype=np.float32)
    one_hot_voice3 = np.zeros((y.shape[0], len(unique_voice3)), dtype=np.float32)
    one_hot_voice4 = np.zeros((y.shape[0], len(unique_voice4)), dtype=np.float32)
    
    # one hot encode each note
    for timestep, notes in enumerate(y):
        for voice, note in enumerate(notes):
            if (voice == 0):
                # get location in uniques of current note
                one_hot_location = uniqueLocation(unique_voice1, note)
                one_hot_voice1[timestep][one_hot_location] = 1
            elif (voice == 1):
                one_hot_location = uniqueLocation(unique_voice2, note)
                one_hot_voice2[timestep][one_hot_location] = 1
            elif (voice == 2):
                one_hot_location = uniqueLocation(unique_voice3, note)
                one_hot_voice3[timestep][one_hot_location] = 1
            elif (voice == 3):
                one_hot_location = uniqueLocation(unique_voice4, note)
                one_hot_voice4[timestep][one_hot_location] = 1

    # print(one_hot_voice1.shape, one_hot_voice2.shape, one_hot_voice3.shape, one_hot_voice4.shape)
    return one_hot_voice1, one_hot_voice2, one_hot_voice3, one_hot_voice4

# set_voices and all_voices used when creating a subset of all data for the current dataset (train/test)
# necessary for one-hot encoding of test data
class NotesDataset(Dataset):
    def __init__(self, window_size: int, subset_voices:np.ndarray, all_voices: np.ndarray):
        # nr of samples, and nr of voices
        self.nr_samples = subset_voices.shape[0] - window_size
        self.nr_voices = subset_voices.shape[1]

        # initialize x data -> window_size amount of notes of 4 voices each per prediction
        self.x = np.zeros((self.nr_samples, window_size, self.nr_voices), dtype=np.float32)
        for i in range(self.x.shape[0]):
            self.x[i] = subset_voices[i : i + window_size]

        # initialize y data -> 4 following target notes per time window 
        self.y = np.zeros((self.nr_samples, self.nr_voices), dtype = np.float32)
        for j in range(self.y.shape[0]):
            self.y[j] = subset_voices[j + window_size]

        # one hot encode different task (differnt voices) target values
        self.y1, self.y2, self.y3, self.y4 = one_hot_encode(self.y, all_voices)

        # create tensors
        self.x = torch.from_numpy(self.x).to(device)
        self.y1 = torch.from_numpy(self.y1).to(device)
        self.y2 = torch.from_numpy(self.y2).to(device)
        self.y3 = torch.from_numpy(self.y3).to(device)
        self.y4 = torch.from_numpy(self.y4).to(device)

    def __getitem__(self, index: int):
        sample = {'x': self.x[index], 'y1': self.y1[index], 'y2': self.y2[index], 'y3': self.y3[index], 'y4': self.y4[index]}
        return sample

    def __len__(self):
        return self.nr_samples
    
# LSTM model with four output heads, one for each voice next note prediction (task)
# The model can be set to stateful, meaning the internal hidden state and cell state is passed
#   into the model each batch and reset once per epoch.
class LSTM_model(nn.Module):
    def __init__(self, input_size, output_sizes, hidden_size, num_layers, batch_size):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # lstm layer(s)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.2, batch_first=True)
        
        # task head: voice 1
        self.head1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(hidden_size, hidden_size)),
             ('relu', nn.ReLU()),
             ('final', nn.Linear(hidden_size, output_sizes[0]))]
        ))

        # task head: voice 2
        self.head2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(hidden_size, hidden_size)),
             ('relu', nn.ReLU()),
             ('final', nn.Linear(hidden_size, output_sizes[1]))]
        ))

        # task head: voice 3
        self.head3 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(hidden_size, hidden_size)),
             ('relu', nn.ReLU()),
             ('final', nn.Linear(hidden_size, output_sizes[2]))]
        ))

        # task head: voice 4
        self.head4 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(hidden_size, hidden_size)),
             ('relu', nn.ReLU()),
             ('final', nn.Linear(hidden_size, output_sizes[3]))]
        ))

        print("LSTM initialized with {} input size, {} hidden layer size, {} number of LSTM layers, and an output size of {}".format(input_size, hidden_size, num_layers, output_sizes))
        # reset states in case of stateless use
        self.reset_states(batch_size)

    # reset hidden state and cell state, should be before each new sequence
    #   In our problem: every epoch, as it is one long sequence
    def reset_states(self, batch_size):
    # def reset_states(self):
        # hidden state and cell state for LSTM 
        self.hn = torch.zeros(self.num_layers,  batch_size, self.hidden_size).to(device)
        self.cn = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def forward(self, input, stateful):
        # simple forward function
        # stateful = keep hidden states entire sequence length
        # only use when 2 samples follow temporally (first timepoint from 2nd sample follows from last timepoint 1st sample)
        if stateful:
            # for last batch which might not be the same shape
            if (input.size(0) != self.hn.size(1)):
                self.reset_states(input.size(0))
              
            # lstm layer
            out, (self.hn, self.cn) = self.lstm(input, (self.hn.detach(), self.cn.detach())) 
            # linear output layers for each head
            task_head1_out = self.head1(out[:,-1,:])
            task_head2_out = self.head2(out[:,-1,:])
            task_head3_out = self.head3(out[:,-1,:])
            task_head4_out = self.head4(out[:,-1,:])
        else:
            # initiaze hidden and cell states
            hn = torch.zeros(self.num_layers,  input.size(0), self.hidden_size).to(device)
            cn = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
            # lstm layer
            out, (hn, cn) = self.lstm(input, (hn, cn))
            # linear output layers for each head
            task_head1_out = self.head1(out[:,-1,:])
            task_head2_out = self.head2(out[:,-1,:])
            task_head3_out = self.head3(out[:,-1,:])
            task_head4_out = self.head4(out[:,-1,:])

        return task_head1_out, task_head2_out, task_head3_out, task_head4_out

def training(model, train_loader:DataLoader, test_loader:DataLoader, nr_epochs, optimizer, loss_func, scheduler, stateful, writer):
    # lowest train/test loss, train/test loss lists
    lowest_train_loss = np.inf
    lowest_test_loss = np.inf
    train_losses = []
    test_losses = []

    # test_loss declaration untill assigned in model evaluation (used in progress bar print)
    test_loss = "n/a"

    # training loop
    for epoch in (progress_bar := tqdm(range(1, nr_epochs))):
        # add epoch info to progress bar
        progress_bar.set_description(f"Epoch {epoch}")

        # reset lstm hidden and cell state (stateful lstm = reset states once per sequence)
        # if not, reset automatically each forward call
        if stateful:
            model.reset_states(train_loader.batch_size)

        # reset running loss
        running_loss_train = 0
        running_loss_test = 0

        # train loop
        model.train()
        for i, data in enumerate(train_loader):
            # reset gradient function of weights
            optimizer.zero_grad()
            # forward
            voice1_pred, voice2_pred, voice3_pred, voice4_pred = model(data["x"], stateful)
            # calculate loss
            loss = loss_func(voice1_pred, data["y1"]) + loss_func(voice2_pred, data["y2"]) + loss_func(voice3_pred, data["y3"]) + loss_func(voice4_pred, data["y4"])
            # backward, retain_graph = True needed for hidden lstm states
            loss.backward(retain_graph=True)
            # step
            optimizer.step()
            # add to running loss
            running_loss_train += loss.item()

        # learning rate scheduler step
        scheduler.step()

        # calc running loss
        train_loss = running_loss_train/len(train_loader)
        train_losses.append(train_loss)

        # add loss to tensorboard
        writer.add_scalar("Running train loss", train_loss, epoch)        

        # check if lowest loss
        if (train_loss < lowest_train_loss):
            lowest_train_loss = train_loss
            # Save model
            torch.save(model.state_dict(), "drive/MyDrive/colab_outputs/lstm_bach/models/model" + str(train_loader.dataset.x.shape[1]) + str(model.hidden_size) + ".pth")

        # Test evaluation
        if (test_loader):
            # model.eval()
            with torch.no_grad():
                for j, data in enumerate(test_loader):
                    # forward pass
                    voice1_pred, voice2_pred, voice3_pred, voice4_pred = model(data["x"], stateful)
                    # calculate loss
                    loss = loss_func(voice1_pred, data["y1"]) + loss_func(voice2_pred, data["y2"]) + loss_func(voice3_pred, data["y3"]) + loss_func(voice4_pred, data["y4"])
                    # add to running loss
                    running_loss_test += loss

            # calc running loss
            test_loss = running_loss_test/len(test_loader)
            test_losses.append(test_loss)

            # add test loss to tensorboard
            writer.add_scalar("Running test loss", test_loss, epoch)

            # if lowest till now, save model (checkpointing)
            if (test_loss < lowest_test_loss):
                lowest_test_loss = test_loss
                torch.save(model.state_dict(), "drive/MyDrive/colab_outputs/lstm_bach/models/model" + str(train_loader.dataset.x.shape[1]) + str(model.hidden_size) + "test" + ".pth")

        # before next epoch: add last epoch info to progress bar
        progress_bar.set_postfix({"train_loss": train_loss, "test_loss": test_loss})

    # save hparams along with lowest train/test losses
    writer.add_hparams(
        {"window_size": train_loader.dataset.x.shape[1], "hidden_size": model.hidden_size},
        {"MinTrainLoss": lowest_train_loss, "MinTestLoss": lowest_test_loss},
    )

    return train_losses, test_losses

# create train and test dataset based on window size where one window of timesteps
#   will predict the subsequential single timestep
# Data is created without any information leak between test/train (either scaling leak or time leak)
def createTrainTestDataloaders(voices, split_size, window_size, batch_size):
    # Train/test split
    dataset_size = len(voices[:,])
    indices = list(range(dataset_size))
    split = int(np.floor((1 - split_size) * dataset_size))
    train_indices, test_indices = indices[:split], indices[split:]

    # create split in data
    train_voices = voices[train_indices, :]
    test_voices = voices[test_indices, :]
    
    # scale both sets, using training data as fit (no leaks)
    scaler = StandardScaler()
    scaler.fit(train_voices)
    train_voices = scaler.transform(train_voices)
    all_voices = scaler.transform(voices)
    
    # create train dataset
    train_dataset = NotesDataset(window_size, train_voices, all_voices)

    # create train dataloader
    train_loader = DataLoader(train_dataset, batch_size)

    # Do the same for test set 
    if (split_size > 0):
        # scale test set
        test_voices = scaler.transform(test_voices)
        # create test dataset
        test_dataset = NotesDataset(window_size, test_voices, all_voices)
        # create test dataloader
        test_loader = DataLoader(test_dataset, batch_size)
    else:
        test_loader = None
    
    return train_loader, test_loader

def main():
    # load data, 4 voices of instruments
    voices = np.loadtxt("input.txt")

    # remove starting silence, does not promote learning
    # data shape is (3816, 4) after
    voices = np.delete(voices, slice(8), axis=0)
    print("Data shape (4 voices):", voices.shape)

    # batch_size for training network
    batch_size = 64

    # split size of test/train data
    split_size = 0.0

    # hyperparameters for fine-tuning
        # window_size = sliding window on time-sequence data for input
        # hidden_size = hidden units of lstm layer(s)
        # conv_channels = number of channels in the first conv layer (multiplied by 2 every next layer)
        # nr_layers = number of lstm layers stacked after each other
    hyperparams = dict(
        window_size = [96],
        hidden_size = [256],
        nr_layers = [2],
        l2 = [0.07]
    )
    # sets of combinations of hparams
    hyperparam_value_sets = product(*[value for value in hyperparams.values()])

    # Loop through different combinations of the hyperparameters
    for run_id, (window_size, hidden_size, nr_layers, l2) in enumerate(hyperparam_value_sets):
        # tensorboard summary writer
        writer = SummaryWriter(f'drive/MyDrive/colab_outputs/lstm_bach/runs/window_size={window_size} hidden_size={hidden_size}')
        
        # Split data in train and test, scale, create datasets and create dataloaders
        train_loader, test_loader = createTrainTestDataloaders(voices, split_size, window_size, batch_size)

        # some informational print statements
        print("\nNew run window/hidden/batch_size:", window_size, "/", hidden_size, "/", batch_size)
        data = next(iter(train_loader))
        print("Input size:", data["x"].size(), 
            "- Output size:[", data["y1"].size(), data["y2"].size(), data["y3"].size(), data["y4"].size(), "]\n",
            "TRAIN batches:", len(train_loader), 
            "- TEST batches:", len(test_loader) if test_loader else "Not available")
        # Input/output dimensions
        input_size = voices.shape[1]
        output_sizes = [data["y1"].size(1), data["y2"].size(1), data["y3"].size(1), data["y4"].size(1)]

        # create model
        lstm_model = LSTM_model(input_size, output_sizes, hidden_size, nr_layers, batch_size)

        # loss function and optimizer
        #   Output of each head is multi-class classification -> cross entropy
        loss_func = nn.CrossEntropyLoss()
        # AdamW = Adam with fixed weight decay (weight decay performed after controlling parameter-wise step size)
        optimizer = optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=l2)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=750)
        
        # to gpu if possible
        lstm_model = lstm_model.to(device)
        
        # training loop
        epochs = 1000
        # In this example we should not use a stateful lstm, as the next samples (subsequent sliding windows) do not follow directly from the current.
        # This is only the case when the first sample is (for Ex.) [1:10] which is the first window, and [11:20] the next, and so on.
        # With our data it would be: [1:10] and the next [2:11]. Target value does not matter necessarily. 
        # More explanation: https://stackoverflow.com/questions/58276337/proper-way-to-feed-time-series-data-to-stateful-lstm
        #   unfortunately I implemented stateful before knowing these in and outs.
        stateful = False
        train_losses, test_losses = training(lstm_model, train_loader, test_loader, epochs, optimizer, loss_func, scheduler, stateful, writer)

        # flush tensorboard writer
        writer.flush()


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True,linewidth=np.nan)

    main()

