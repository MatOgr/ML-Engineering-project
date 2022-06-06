from model import AvocadoDataset, evaluate_model
from torch.utils.data import DataLoader
from torch.jit import load as load_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


# * Load the test data
test_data = DataLoader(AvocadoDataset(
    './data/avocado.data.test'), batch_size=1, shuffle=False)

# * Load the model
model = load_model('./data/model_scripted.pt')
model.eval()

# * Append new inference data
with open('./data/evaluation_results.csv', 'a+') as f:
    f.write("{0},{1},{2}\n".format(*evaluate_model(test_data, model)))

# * Load all inference data gathered (till the current one)
results = pd.read_csv('./data/evaluation_results.csv',
                      names=['MSE', 'RMSE', 'MAE'])

with open('logs.md', 'w') as f:
    f.write("MSE: {0}, RMSE: {1}, MAE: {2}\n".format(
        results['MSE'], results['RMSE'], results['MAE']))

# * Plot the results
plt.plot(range(1, len(results)+1), results['MSE'], color='green')
plt.scatter(range(1, len(results)+1),
            results['MSE'], label='MSE', color='green', marker='.')
plt.plot(range(1, len(results)+1), results['RMSE'], color='darkred')
plt.scatter(range(1, len(results)+1),
            results['RMSE'], label='RMSE', color='darkorange', marker='.')
plt.plot(range(1, len(results)+1), results['MAE'], color='blue')
plt.scatter(range(1, len(results)+1),
            results['MAE'], label='MAE', color='blue', marker='.')
plt.xticks(range(1, len(results)+1))
plt.ylabel('Metric value')
plt.xlabel('Build number')
plt.legend()

# * Save figure
plt.savefig('data/plots.png')
