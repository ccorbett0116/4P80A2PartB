import matplotlib.pyplot as plt
import numpy as np
from MultiLayerFFNN import MLFFNN
import multiprocessing
from main import read_data

def train_network(args):
    input_size, hidden_layers, hidden_sizes, output_size, features, labels, num_epochs, lr, loss, features_test, labels_test, idx, seed = args
    nn = MLFFNN(input_size, hidden_layers, hidden_sizes, output_size, features, labels, num_epochs, lr, loss, features_test, labels_test, idx, seed)
    nn.train(idx)
    return idx, nn

def main():
    lr = 0.0008125
    epochs = 2200
    seeds = []
    np.random.seed(7246325)
    runs = 30
    train_args = []
    for i in range(runs):
        seeds.append(np.random.randint(0, 10000000))

    for i in range(runs):
        labels, features = read_data("L30fft1000.out")
        test_size = (1 / 3)
        train_size = int(len(labels) * (1 - test_size))
        np.random.seed(seeds[i])
        shuffled_indices = np.random.permutation(len(labels))
        labels = np.array(labels, dtype=int)[shuffled_indices]
        features = np.array(features, dtype=float)[shuffled_indices]
        labels, labels_test = labels[:train_size], labels[train_size:]
        features, features_test = features[:train_size], features[train_size:]
        input_size = features.shape[1]
        hidden_size = int(input_size * 1)
        output_size = 2
        train_args.append((input_size, 1, [hidden_size], output_size, features, labels, epochs, lr, "MSE", features_test,
                           labels_test, i, seeds[i]))
    batch_size = 20
    neural_networks = []
    for i in range(0, len(train_args), batch_size):
        with multiprocessing.Pool(processes=(batch_size)) as pool:
            results = pool.map(train_network, train_args[i:i + batch_size])
            results.sort(key=lambda x: x[0])
        neural_networks.extend([nn for _, nn in results])
    neural_networks.sort(key=lambda x: x.idx)

    avgTrainPerformance = np.zeros(epochs)
    avgTestPerformance = np.zeros(epochs)
    avgCorrectTraining = 0
    avgIncorrectTraining = 0
    avgCorrectTest = 0
    avgIncorrectTest = 0
    for nn in neural_networks:
        avgTrainPerformance += np.array(nn.trainingProgress)
        avgTestPerformance += np.array(nn.testPerformance)
        for i in range(len(nn.input_data)):
            raw_output, _ = nn.forward_pass(nn.input_data[i])
            if np.argmax(raw_output) == nn.labels[i]:
                avgCorrectTraining += 1
            else:
                avgIncorrectTraining += 1
        for i in range(len(nn.features_test)):
            raw_output, _ = nn.forward_pass(nn.features_test[i])
            if np.argmax(raw_output) == nn.labels_test[i]:
                avgCorrectTest += 1
            else:
                avgIncorrectTest += 1

    avgTrainPerformance /= runs
    avgTestPerformance /= runs
    avgCorrectTest /= runs
    avgCorrectTraining /= runs
    avgIncorrectTest /= runs
    avgIncorrectTraining /= runs
    print("Average Training Performance: ", avgCorrectTraining / (avgCorrectTraining + avgIncorrectTraining))
    plt.plot(avgTrainPerformance)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Training Loss")
    plt.savefig("avgTrainLoss.png")
    plt.clf()
    print("Average Test Performance: ", avgCorrectTest / (avgCorrectTest + avgIncorrectTest))
    plt.plot(avgTestPerformance)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Test Loss")
    plt.savefig("avgTestLoss.png")
    plt.clf()




if __name__ == "__main__":
    main()