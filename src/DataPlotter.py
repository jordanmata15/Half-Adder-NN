import matplotlib.pyplot as plt
import os
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
PLOT_DIR = os.path.join(SRC_DIR, "plots")
TRAINING_LOGS_DIR = os.path.join(SRC_DIR, "training_history")


def plot_data(data_df):
    data_df = data_df.set_index('epoch')\
                     .rename(columns={'mse': 'Mean Square Error'})
    mse_df = data_df[['Mean Square Error']]
    mse_df.plot()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.tight_layout()
    plt.grid()
    #plt.show()
    plt.savefig(os.path.join(PLOT_DIR, 'plot.png'))

if __name__ == "__main__":
    training_log_file = os.path.join(TRAINING_LOGS_DIR, "history.csv")
    training_log_df = pd.read_csv(training_log_file)
    plot_data(training_log_df)