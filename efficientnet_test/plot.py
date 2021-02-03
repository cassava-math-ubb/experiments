import matplotlib.pyplot as plt
from numpy import genfromtxt


def generate_plot(filename, fold):
    my_data = genfromtxt(filename, delimiter=' ')

    # summarize history for accuracy
    accuracy = my_data[:, 10]
    val_accuracy = my_data[:, 16]
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'k_fold_plots/accuracy_{fold}.png')
    plt.close()

    # summarize history for loss
    loss = my_data[:, 7]
    val_loss = my_data[:, 13]
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'k_fold_plots/loss_{fold}.png')
    plt.close()


if __name__ == '__main__':
    generate_plot('k_fold_plots/logs_1.csv', 1)
    generate_plot('k_fold_plots/logs_2.csv', 2)
    generate_plot('k_fold_plots/logs_3.csv', 3)
    generate_plot('k_fold_plots/logs_4.csv', 4)
    generate_plot('k_fold_plots/logs_5.csv', 5)
