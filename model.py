import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
from torch import nn
from torch.utils import data as t_u_data

print(
    f"PyTorch working?\t →\t{torch.__version__}\nLooks like potatoe...but seems to be fine")


# * Customized Dataset class (base provided by PyTorch)
class AvocadoDataset(t_u_data.Dataset):
    def __init__(self, path: str, target: str = 'AveragePrice'):
        data = pd.read_csv(path)
        self.y = data.values[:, 1].astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        self.x_shape = data.drop([target], axis=1).shape
        self.x_data = data.drop(
            [target], axis=1).values.astype('float32')
        # print("Data shape is: ", self.x_data.shape)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return [self.x_data[idx], self.y[idx]]

    def get_shape(self):
        return self.x_shape

    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self.x_data))
        train_size = len(self.x_data) - test_size
        return t_u_data.random_split(self, [train_size, test_size])


class AvocadoRegressor(nn.Module):
    def __init__(self, input_dim):
        super(AvocadoRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 32)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(32, 8)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(8, 1)
        nn.init.xavier_uniform_(self.hidden3.weight)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        return x


def prepare_data(path):
    dataset = AvocadoDataset(path)
    train, test = dataset.get_splits()
    train_dl = t_u_data.DataLoader(train, batch_size=32, shuffle=True)
    test_dl = t_u_data.DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


def train_model(train_dl, model, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    to_compare = None

    for epoch in range(epochs):
        if epoch == 0:
            print(f"Epoch: {epoch+1}")
        if epoch > 0 and (epoch+1) % 10 == 0:
            print(
                f"Epoch: {epoch+1}\tloss\t→\t{mean_squared_error(to_compare[1].detach().numpy(), to_compare[0].detach().numpy())}")
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            # * For loss value inspection
            to_compare = (yhat, targets)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()


def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for _, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        # * retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # * store predictions
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # * return MSE value
    mse = mean_squared_error(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions, squared=False)
    mae = mean_absolute_error(actuals, predictions)
    return mse, rmse, mae


def predict(row, model):
    row = row[0].flatten()
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat


if __name__ == '__main__':

    # * Model parameters
    parser = argparse.ArgumentParser(description="Script performing logistic regression model training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-e", "--epochs", default=100, help="Number of epochs the model will be trained for")
    parser.add_argument("--save", action="store_true",
                        help="Save trained model to file 'trained_model.h5'")

    args = vars(parser.parse_args())

    epochs = args['epochs']
    save_model = args['save']
    print(
        f"Your model will be trained for {epochs} epochs. Trained model will {'not ' if save_model else ''}be saved.")

    # * Paths to data
    avocado_train = './data/avocado.data.train'
    avocado_valid = './data/avocado.data.valid'
    avocado_test = './data/avocado.data.test'

    # * Data preparation
    train_dl = t_u_data.DataLoader(AvocadoDataset(
        avocado_train), batch_size=32, shuffle=True)
    validate_dl = t_u_data.DataLoader(AvocadoDataset(
        avocado_valid), batch_size=128, shuffle=True)
    test_dl = t_u_data.DataLoader(AvocadoDataset(
        avocado_test), batch_size=1, shuffle=False)
    print(f"""
          Train set size: {len(train_dl.dataset)},
          Validate set size: {len(validate_dl.dataset)}
          Test set size: {len(test_dl.dataset)}
          """)

    # * Model definition
    # ! 66 - in case only regions and type are used (among all the categorical vals)
    model = AvocadoRegressor(235)

    # * Train model
    print("Let's start the training, mate!")
    train_model(train_dl, model, int(epochs))

    # * Evaluate model
    mse, rmse, mae = evaluate_model(validate_dl, model)
    print(f"\nEvaluation\t→\tMSE: {mse}, RMSE: {rmse}, MAE: {mae}")

    # * Prediction
    predictions = [(predict(row, model)[0], row[1].item()) for row in test_dl]
    preds_df = pd.DataFrame(predictions, columns=["Prediction", "Target"])
    print("\nNow predictions - hey ho, let's go!\n",
          preds_df.head(), "\n\n...let's save them\ndum...\ndum...\ndum dum dum...\n\tDUM\n")
    preds_df.to_csv("./data/predictions.csv", index=False)

    # * Save the trained model
    if save_model:
        print("Your model has been saved - have a nice day!")
        scripted_model = torch.jit.script(model)
        scripted_model.save('./data/model_scripted.pt')
