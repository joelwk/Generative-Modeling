import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

def extract_date_parts(data, column_name):
    data['hour'] = pd.to_datetime(data[column_name]).dt.hour
    data['day'] = pd.to_datetime(data[column_name]).dt.day
    data['month'] = pd.to_datetime(data[column_name]).dt.month

def plot_kde(data, w, h):
    if 'posted_date_time' not in data.columns:
        print("Column 'posted_date_time' not found in data.")
    else:
        extract_date_parts(data, 'posted_date_time')
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(w, h))
        sns.kdeplot(data['hour'], fill=True, ax=ax[0])
        sns.kdeplot(data['day'], fill=True, ax=ax[1])
        sns.kdeplot(data['month'], fill=True, ax=ax[2])
        for i, label in enumerate(['Hour', 'Day', 'Month']):
            ax[i].set_title(f'Distribution of Rows by {label}')
            ax[i].set_xlabel(label)
            ax[i].set_ylabel('Density')
        plt.tight_layout()
        plt.show()

def plot_hist(data, w, h):
    if 'posted_date_time' not in data.columns:
        print("Column 'posted_date_time' not found in data.")
        return

    extract_date_parts(data, 'posted_date_time')
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(w, h))

    # Plot distribution of rows by hour
    sns.histplot(data['hour'], kde=True, bins=24, ax=ax[0])
    ax[0].set_title('Density Plot for Hour')
    ax[0].set_xlabel('Hour')
    ax[0].set_ylabel('Density')

    # Plot distribution of rows by day
    sns.histplot(data['day'], kde=True, bins=31, ax=ax[1])
    ax[1].set_title('Density Plot for Day')
    ax[1].set_xlabel('Day')
    ax[1].set_ylabel('Density')

    # Plot distribution of rows by month
    sns.histplot(data['month'], kde=True, bins=12, ax=ax[2])
    ax[2].set_title('Density Plot for Month')
    ax[2].set_xlabel('Month')
    ax[2].set_ylabel('Density')
    # Display the figure with subplots
    plt.tight_layout()
    plt.show()

def profile_date_plot(df, date_col):
    df = df.copy()
    # Convert date_col to datetime
    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d %H:%M:%S')
    # Print the minimum and maximum dates
    print(f"Minimum date: {df[date_col].min()}")
    print(f"Maximum date: {df[date_col].max()}")
    # Create a histogram of the dates
    df[date_col].hist(bins=50, figsize=(10, 5))
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dates')
    plt.show()

# Plot the accuracy and loss
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Plot training and validation accuracy
    ax1 = axes[0]
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Training and Validation Accuracy')
    ax1.grid(True)

    ax1_val = ax1.twinx() # Create a secondary y-axis
    ax1_val.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax1_val.set_ylabel('Validation Accuracy', color='red')
    ax1_val.tick_params(axis='y', labelcolor='red')

    # Plot training and validation loss
    ax2 = axes[1]
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_title('Training and Validation Loss')
    ax2.grid(True)

    ax2_val = ax2.twinx() # Create a secondary y-axis
    ax2_val.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax2_val.set_ylabel('Validation Loss', color='red')
    ax2_val.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.show()
