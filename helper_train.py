from typing import Any, Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from tqdm import tqdm


def plot_losses(train_losses: Iterable[float], train_accuracy: Iterable[float],
                val_losses: Iterable[float], val_accuracy: Iterable[float]):
    """Plot losses and accuracy while training.

    Args:
        train_losses (Iterable[float]): sequence of train losses
        train_accuracy (Iterable[float]): sequence of train accuracy
        val_losses (Iterable[float]): sequence of validation losses
        val_accuracy (Iterable[float]): sequence of validation accuracy
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[1].plot(range(1, len(train_accuracy) + 1), train_accuracy, label='train')
    axs[1].plot(range(1, len(val_accuracy) + 1), val_accuracy, label='val')

    if max(train_losses) / min(train_losses) > 10:
        axs[0].set_yscale('log')

    if max(train_accuracy) / min(train_accuracy) > 10:
        axs[0].set_yscale('log')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    axs[0].set_ylabel('loss')
    axs[1].set_ylabel('metric')
    plt.show()


def train_cifar10_classifier(model: Any, optimizer: Any, criterion: Any, accuracy: Callable,
                             train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader | None,
                             num_epochs: int, device: torch.device, verbose: bool = True) -> Tuple[List[float]]:
    """Train and validate neural network

    Args:
        model (Any): neural network to train
        optimizer (Any): optimizer chained to a model
        criterion (Any): loss function
        accuracy (Callable): function to mesure accuracy
        train_loader (torch.utils.data.DataLoader): train DataLoader
        val_loader (torch.utils.data.DataLoader | None): validation DataLoader
        num_epochs (int): number of epochs to train through
        device (torch.device): device to perform computation on
        verbose (bool, optional): whether to verbose the process or not. Defaults to True.

    Returns:
        Tuple[List[float]]: train and validation metrics
    """
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []

    model.to(device)

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss, running_metric = 0, 0
        pbar = tqdm(train_loader, desc=f'Training {epoch}/{num_epochs}') \
            if verbose else train_loader

        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predict = model(X_batch)

            loss = criterion(predict, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric_value = accuracy(predict, y_batch)
                if metric_value is torch.Tensor:
                    metric_value = metric_value.item()
                running_loss += loss.item() * y_batch.size(0)
                running_metric += metric_value * y_batch.size(0)

            if verbose and i % 100 == 0:
                pbar.set_postfix({'loss': loss.item(), 'accuracy': metric_value})

        train_losses += [running_loss / len(train_loader.dataset)]
        train_metrics += [running_metric / len(train_loader.dataset)]

        if val_loader:
            model.eval()
            running_loss, running_metric = 0, 0
            pbar = tqdm(val_loader, desc=f'Validating {epoch}/{num_epochs}') \
                if verbose else val_loader

            for i, (X_batch, y_batch) in enumerate(pbar, 1):
                with torch.no_grad():
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    predict = model(X_batch)
                    loss = criterion(predict, y_batch)

                    metric_value = accuracy(predict, y_batch)
                    if metric_value is torch.Tensor:
                        metric_value = metric_value.item()
                    running_loss += loss.item() * y_batch.size(0)
                    running_metric += metric_value * y_batch.size(0)

                if verbose and i % 100 == 0:
                    pbar.set_postfix({'loss': loss.item(), 'accuracy': metric_value})

            val_losses += [running_loss / len(val_loader.dataset)]
            val_metrics += [running_metric / len(val_loader.dataset)]

        if verbose:
            plot_losses(train_losses, train_metrics, val_losses, val_metrics)

    return train_metrics, val_metrics


def metric_accuracy(y_pred: Iterable[float], y_true: Iterable[float]):
    """Function to measure accuracy while training.

    Args:
        y_pred (Iterable[float]): predicted probabilities of the classes
        y_true (Iterable[float]): ground truth classes

    Returns:
        _type_: _description_
    """
    _, y_pred = torch.max(y_pred, dim=1)
    return (y_pred == y_true).sum().item() / y_true.size(0)


def eval_cifar10_classifier(model: Any, test_loader: torch.utils.data.DataLoader,
                            classes: List[str], device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Evaluate total accuracy of the trained model and accuracy for each class.

    Args:
        model (Any): model to evaluate
        test_loader (torch.utils.data.DataLoader): test DataLoader
        classes (List[str]): list of the classes
        device (torch.device): device to perform computations on

    Returns:
        Tuple[float, Dict[str, float]]: total accuracy and accuracies for each class
    """
    model.eval()
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    total_acc = 100 * sum(correct_pred.values()) / sum(total_pred.values())
    print(f'Total accuracy: {total_acc:.2f} %')

    acc_for_class = {classname: 0 for classname in classes}

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        acc_for_class[classname] = accuracy
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    return total_acc, acc_for_class
