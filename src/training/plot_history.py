import matplotlib.pyplot as plt
import pandas as pd


def plot_history(train_loss_file, val_loss_file, train_accuracy_file,
                 val_accuracy_file):
    train_loss = pd.read_csv(train_loss_file)
    val_loss = pd.read_csv(val_loss_file)
    train_accuracy = pd.read_csv(train_accuracy_file)
    val_accuracy = pd.read_csv(val_accuracy_file)
    fig, (loss, accuracy) = plt.subplots(2, 1)
    loss.set_xlabel('Epoka')
    loss.set_ylabel('Funkcja kosztu')
    accuracy.set_xlabel('Epoka')
    accuracy.set_ylabel('Dokładność klasyfikacji')
    loss.plot(train_loss['Value'], label='Zbiór uczący')
    loss.plot(val_loss['Value'], label='Zbiór walidacyjny')
    accuracy.plot(train_accuracy['Value'], label='Zbiór uczący')
    accuracy.plot(val_accuracy['Value'], label='Zbiór walidacyjny')
    loss.legend()
    accuracy.legend()
    loss.grid()
    accuracy.grid()
    plt.show()
