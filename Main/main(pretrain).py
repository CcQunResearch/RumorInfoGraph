# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 16:01
# @Author  :
# @Email   :
# @File    : main(pretrain).py
# @Software: PyCharm
# @Note    :
import sys
import os.path as osp

dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Main.pargs import pargs
from Main.utils import create_log_dict_pretrain, write_log, write_json
from Main.sort import sort_weibo_dataset, sort_weibo_self_dataset, sort_weibo_2class_dataset
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.dataset import WeiboDataset
from Main.model import Net
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def pre_train(unsup_train_loader, model, optimizer, separate_encoder, lamda, device):
    model.train()
    loss_all = 0

    for data in unsup_train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        unsup_loss = model.unsup_loss(data)
        if separate_encoder:
            unsup_sup_loss = model.unsup_sup_loss(data)
            loss = unsup_loss + unsup_sup_loss * lamda
        else:
            loss = unsup_loss

        loss.backward()
        optimizer.step()

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(unsup_train_loader.dataset)


def fine_tuning(train_loader, model, optimizer, device):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        loss = F.binary_cross_entropy(model(data), data.y.to(torch.float32))
        loss.backward()

        loss_all += loss.item() * data.num_graphs

        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(model, dataloader, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []
    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        error += F.binary_cross_entropy(pred, data.y.to(torch.float32)).item() * data.num_graphs
        y_true += data.y.tolist()
        y_pred += pred.tolist()
    acc = accuracy_score(y_true, y_pred)
    prec = [precision_score(y_true, y_pred, pos_label=1, average='binary'),
            precision_score(y_true, y_pred, pos_label=0, average='binary')]
    rec = [recall_score(y_true, y_pred, pos_label=1, average='binary'),
           recall_score(y_true, y_pred, pos_label=0, average='binary')]
    f1 = [f1_score(y_true, y_pred, pos_label=1, average='binary'),
          f1_score(y_true, y_pred, pos_label=0, average='binary')]
    return error / len(dataloader.dataset), acc, prec, rec, f1


def test_and_log(model, val_loader, test_loader, device, epoch, lr, loss, train_acc, ft_log_record):
    val_error, val_acc, val_prec, val_rec, val_f1 = test(model, val_loader, device)
    test_error, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Train ACC: {:.3f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
        .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc, test_prec[0], test_prec[1],
                test_rec[0],
                test_rec[1], test_f1[0], test_f1[1])

    ft_log_record['val accs'].append(round(val_acc, 3))
    ft_log_record['test accs'].append(round(test_acc, 3))
    ft_log_record['test prec T'].append(round(test_prec[0], 3))
    ft_log_record['test prec F'].append(round(test_prec[1], 3))
    ft_log_record['test rec T'].append(round(test_rec[0], 3))
    ft_log_record['test rec F'].append(round(test_rec[1], 3))
    ft_log_record['test f1 T'].append(round(test_f1[0], 3))
    ft_log_record['test f1 F'].append(round(test_f1[1], 3))
    return val_error, log_info, ft_log_record


if __name__ == '__main__':
    args = pargs()

    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    ft_runs = args.ft_runs

    batch_size = args.batch_size
    unsup_bs_ratio = args.unsup_bs_ratio
    weight_decay = args.weight_decay
    epochs = args.epochs
    ft_epochs = args.ft_epochs

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', unsup_dataset, 'dataset')
    model_path = osp.join(dirname, '..', 'Model', f'w2v_{dataset}_{unsup_train_size}_{vector_size}.model')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')
    weight_path = osp.join(dirname, '..', 'Model', f'{log_name}.pt')

    log = open(log_path, 'w')
    log_dict = create_log_dict_pretrain(args)

    if not osp.exists(model_path):
        if dataset == 'Weibo':
            sort_weibo_dataset(label_source_path, label_dataset_path)
        elif 'DRWeibo' in dataset:
            sort_weibo_2class_dataset(label_source_path, label_dataset_path)

        sentences = collect_sentences(label_dataset_path, unlabel_dataset_path, unsup_train_size)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)

    word2vec = Embedding(model_path)

    for run in range(runs):
        unlabel_dataset = WeiboDataset(unlabel_dataset_path, word2vec, clean=False)
        unsup_train_loader = DataLoader(unlabel_dataset, batch_size * unsup_bs_ratio, shuffle=True)

        model = Net(vector_size, args.hidden, args.separate_encoder).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

        write_log(log, f'runs:{run}')
        log_record = {
            'run': run,
            'record': []
        }

        for epoch in range(1, epochs + 1):
            pretrain_loss = pre_train(unsup_train_loader, model, optimizer, args.separate_encoder, args.lamda, device)

            log_info = 'Epoch: {:03d}, Loss: {:.7f}'.format(epoch, pretrain_loss)
            write_log(log, log_info)

        torch.save(model.state_dict(), weight_path)
        write_log(log, '')

        ks = [10, 20, 40, 80, 100, 200, 300, 500, 10000]
        for k in ks:
            for r in range(ft_runs):
                ft_lr = args.ft_lr
                write_log(log, f'k:{k}, r:{r}')

                ft_log_record = {'k': k, 'r': r, 'val accs': [], 'test accs': [], 'test prec T': [], 'test prec F': [],
                                 'test rec T': [], 'test rec F': [], 'test f1 T': [], 'test f1 F': []}

                if dataset == 'Weibo':
                    sort_weibo_dataset(label_source_path, label_dataset_path, k_shot=k)
                elif 'DRWeibo' in dataset:
                    sort_weibo_2class_dataset(label_source_path, label_dataset_path, k_shot=k)

                train_dataset = WeiboDataset(train_path, word2vec)
                val_dataset = WeiboDataset(val_path, word2vec)
                test_dataset = WeiboDataset(test_path, word2vec)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                model.load_state_dict(torch.load(weight_path))
                optimizer = Adam(model.parameters(), lr=args.ft_lr, weight_decay=weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

                val_error, log_info, ft_log_record = test_and_log(model, val_loader, test_loader,
                                                                  device, 0, args.ft_lr, 0, 0, ft_log_record)
                write_log(log, log_info)

                for epoch in range(1, ft_epochs + 1):
                    ft_lr = scheduler.optimizer.param_groups[0]['lr']
                    _ = fine_tuning(train_loader, model, optimizer, device)

                    train_error, train_acc, _, _, _ = test(model, train_loader, device)
                    val_error, log_info, ft_log_record = test_and_log(model, val_loader, test_loader, device, epoch,
                                                                      ft_lr, train_error, train_acc, ft_log_record)
                    write_log(log, log_info)

                    scheduler.step(val_error)

                ft_log_record['mean acc'] = round(np.mean(ft_log_record['test accs'][-10:]), 3)
                log_record['record'].append(ft_log_record)
                write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
