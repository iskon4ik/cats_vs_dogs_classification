import torch

from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_step(model, data, labels, criterion, optimizer, scheduler, device):
    data, labels = data.to(device), labels.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def val_step(model, data, labels, criterion, optimizer, scheduler, device):
    data, labels = data.to(device), labels.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    pred = torch.argmax(output, dim=1).cpu().numpy()
    return loss.item(), pred, labels.cpu().numpy()


def print_epoch_metrics(batch_loss_train, batch_loss_val, dl_train, dl_val,
                        cls_size, cls_true_cnt,
                        y_true, y_pred):
    print('Train mean loss:', batch_loss_train / len(dl_train), ', Val mean loss:', batch_loss_val / len(dl_val))
    print('Classwise accuracy:')
    s = []
    for cls in cls_size.keys():
        acc = cls_true_cnt[cls] / cls_size[cls]
        s.append(f'Class {cls} - {acc}')
    print(', '.join(s))
    print(classification_report(y_true, y_pred))
    print('-' * 40)


def train_val(model, dl_train, dl_val, epochs, criterion, optimizer, scheduler, device):
    loss_train = []
    loss_val = []
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        model.train()
        batch_loss_train = 0
        for data, labels in tqdm(dl_train):
            loss_value = train_step(model, data, labels, criterion, optimizer, scheduler, device)
            batch_loss_train += loss_value

        batch_loss_val = 0
        model.eval()
        cls_size = {0: 0, 1: 0}
        cls_true_cnt = {0: 0, 1: 0}
        y_pred = []
        y_true = []
        for data, labels in tqdm(dl_val):
            loss_value, pred, labels_cpu = val_step(model, data, labels, criterion,
                                                    optimizer, scheduler, device)
            batch_loss_val += loss_value
            y_pred += list(pred)
            y_true += list(labels_cpu)
            for p, y in zip(pred, labels_cpu):
                cls_size[y] += 1
                if p == y:
                    cls_true_cnt[y] += 1

        loss_train.append(batch_loss_train / len(dl_train))
        loss_val.append(batch_loss_val / len(dl_val))

        print_epoch_metrics(
            batch_loss_train, batch_loss_val, dl_train, dl_val, cls_size, cls_true_cnt,
            y_true, y_pred
        )

    return loss_train, loss_val


def plot_losses(loss_train, loss_val):
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.legend()
    plt.title('training and validation loss')
    plt.show()
