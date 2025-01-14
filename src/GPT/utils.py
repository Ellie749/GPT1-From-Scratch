import matplotlib.pyplot as plt


def plot_metrics(history: list) -> None:
    plt.plot(range(len(history)), history, label="loss")
    plt.title("Loss history")
    plt.show()
    plt.savefig("GPT/loss_history.png")
