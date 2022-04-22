from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from Utils import Embedding
from Main import WeiboDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")


def train(epoch, use_unsup_loss):
    model.train()
    loss_all = 0
    sup_loss_all = 0
    unsup_loss_all = 0
    unsup_sup_loss_all = 0

    if use_unsup_loss:
        for data, data2 in zip(train_loader, unsup_train_loader):
            data = data.to(device)
            data2 = data2.to(device)
            optimizer.zero_grad()

            sup_loss = F.binary_cross_entropy(model(data), data.y.to(torch.float32))
            unsup_loss = model.unsup_loss(data2) + model.unsup_loss(data)
            if separate_encoder:
                unsup_sup_loss = model.unsup_sup_loss(data2) + model.unsup_sup_loss(data)
                loss = sup_loss + unsup_loss + unsup_sup_loss * lamda
            else:
                loss = sup_loss + unsup_loss * lamda

            loss.backward()

            sup_loss_all += sup_loss.item()
            unsup_loss_all += unsup_loss.item()
            if separate_encoder:
                unsup_sup_loss_all += unsup_sup_loss.item()
            loss_all += loss.item() * data.num_graphs

            optimizer.step()

        if separate_encoder:
            print(sup_loss_all, unsup_loss_all, unsup_sup_loss_all)
        else:
            print(sup_loss_all, unsup_loss_all)
        return loss_all / len(train_loader.dataset)
    else:
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            sup_loss = F.binary_cross_entropy(model(data), data.y)
            loss = sup_loss

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    correct_num = 0
    y_true = []
    y_pred = []
    for data in loader:
        # print(data.num_graphs)
        # print(data.num_nodes)
        # print(data.x.shape)
        # print(data.y)
        # print(data.edge_index.shape)
        # print(data.edge_attr.shape)
        # print(list(data.edge_index.data.numpy()))
        data = data.to(device)
        pred = model(data)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        y_true += data.y.tolist()
        y_pred += pred.tolist()
        error += F.binary_cross_entropy(pred, data.y.to(torch.float32)).item() * data.num_graphs
    acc = accuracy_score(y_true, y_pred)
    prec = [precision_score(y_true, y_pred, pos_label=1, average='binary'),
            precision_score(y_true, y_pred, pos_label=0, average='binary')]
    rec = [recall_score(y_true, y_pred, pos_label=1, average='binary'),
           recall_score(y_true, y_pred, pos_label=0, average='binary')]
    f1 = [f1_score(y_true, y_pred, pos_label=1, average='binary'),
          f1_score(y_true, y_pred, pos_label=0, average='binary')]
    return error / len(loader.dataset), acc, prec, rec, f1


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed_everything()
    from Main.model import Net
    from Main.arguments import arg_parse

    # ============
    # Hyperparameters
    # ============
    args = arg_parse()
    target = args.target
    dim = 64
    epochs = 500
    batch_size = 8
    lamda = args.lamda
    use_unsup_loss = args.use_unsup_loss
    separate_encoder = args.separate_encoder

    dirname = os.path.dirname(os.path.abspath(__file__))
    label_path = os.path.join(dirname, '..', 'Data', 'Weibo', 'processed', 'label')
    unlabel_path = os.path.join(dirname, '..', 'Data', 'Weibo', 'processed', 'unlabel')
    model_path = os.path.join(dirname, '..', 'Model', 'w2v.model')

    word2vec = Embedding(model_path)

    label_dataset = WeiboDataset(label_path, word2vec)
    unlabel_dataset = WeiboDataset(unlabel_path, word2vec, clean=False)

    train_dataset = label_dataset[:int(len(label_dataset) * 0.8)]
    val_dataset = label_dataset[int(len(label_dataset) * 0.8):int(len(label_dataset) * 0.9)]
    test_dataset = label_dataset[int(len(label_dataset) * 0.9):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    if use_unsup_loss:
        unsup_train_loader = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=True)

        print(len(train_dataset), len(val_dataset), len(test_dataset), len(unlabel_dataset))
    else:
        print(len(train_dataset), len(val_dataset), len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model = Net(word2vec.embedding_dim, dim, use_unsup_loss, separate_encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

    val_error, val_acc, val_prec, val_rec, val_f1 = test(val_loader)
    test_error, test_acc, test_prec, test_rec, test_f1 = test(test_loader)
    print(
        'Epoch: {:03d}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, '
        'Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
            .format(0, val_error, test_error, val_acc, test_acc, test_prec[0], test_prec[1], \
                    test_rec[0], test_rec[1], test_f1[0], test_f1[1]))

    best_val_error = None
    for epoch in range(1, epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch, use_unsup_loss)
        val_error, val_acc, val_prec, val_rec, val_f1 = test(val_loader)
        scheduler.step(val_error)

        # if best_val_error is None or val_error <= best_val_error:
        #     test_error, test_acc = test(test_loader)
        #     best_val_error = val_error
        test_error, test_acc, test_prec, test_rec, test_f1 = test(test_loader)

        print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, '
              'Test BCE: {:.7f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, '
              'Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
              .format(epoch, lr, loss, val_error, test_error, val_acc, test_acc, test_prec[0], test_prec[1], \
                      test_rec[0], test_rec[1], test_f1[0], test_f1[1]))

    with open('supervised.log', 'a+') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(target, args.train_num, use_unsup_loss, separate_encoder, args.lamda,
                                                   args.weight_decay, val_error, test_error))
