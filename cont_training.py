# this file is dedicated to continue the training of a model if stopped before it has finished.
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from LSTM_bach import LSTM_model, NotesDataset, training, createTrainTestDataloaders



def main():
    # if you have an GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define parameters used when model was initialized here
    window_size = 80
    hidden_size = 256
    conv_channels = 8
    num_layers = 2
    batch_size = 1
    # train/test split
    split_size = 0.1
    # non-variable parameters for bach music
    input_size = 4
    output_size = 98

    # initialize model
    model = LSTM_model(input_size, output_size, hidden_size, num_layers, batch_size, conv_channels)
    # load model weights
    model.load_state_dict(torch.load("models/model802568.pth", map_location=torch.device('cpu')))
    model.reset_states(batch_size)

    # load data, 4 voices of instruments
    voices = np.loadtxt("input.txt")
    # remove starting silence, does not promote learning
    # data shape is (3816, 4) after
    voices = np.delete(voices, slice(8), axis=0)

    # create train/test dataloader
    train_loader, test_loader = createTrainTestDataloaders(voices, split_size, window_size, batch_size)

    # loss function and optimizer
    #   multi lable one hot encoded prediction only works with BCEwithlogitloss
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # to gpu if possible
    model = model.to(device)

    # tensorboard summary writer
    writer = SummaryWriter(f'runs/window_size{window_size}_hidden_size{hidden_size}_conv_channels{conv_channels}_cont')

    # training loop
    epochs = 400
    stateful = True
    lowest_train_loss, lowest_test_loss = training(model, train_loader, test_loader, epochs, optimizer, loss_func, stateful, writer)

    # save hparams along with lowest train/test losses
    writer.add_hparams(
        {"window_size": window_size, "hidden_size": hidden_size, "conv_channels": conv_channels},
        {"MinTrainLoss": lowest_train_loss, "MinTestLoss": lowest_test_loss},
    )
    # tb writer flush
    writer.flush()

if __name__ == '__main__':
    main()