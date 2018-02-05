# -*- coding:utf8 -*-
import numpy as np
from chainer import Variable, optimizers
from sklearn.cross_validation import train_test_split
import net
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pyltr
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_FILEPATH_ALL = '/home/kongxiangfei/workspaces/pycharm_workspaces/eventranker/datasets/all.txt'
FEATURE_FILEPATH_TRAIN = '/home/kongxiangfei/workspaces/pycharm_workspaces/eventranker/datasets/train.txt'
# FEATURE_FILEPATH_TRAIN = 'E:/workspace/pycharm_projects/lunwen/OPEREVENT/eventranker/datasets/train.txt'
FEATURE_FILEPATH_TEST = '/home/kongxiangfei/workspaces/pycharm_workspaces/eventranker/datasets/test.txt'
# FEATURE_FILEPATH_TEST = 'E:/workspace/pycharm_projects/lunwen/OPEREVENT/eventranker/datasets/test.txt'


# y ~ [1, n_rank] x ~ N(x|w * y, sigma)
def make_dataset(n_dim, n_rank, n_sample, sigma):
    ys = np.random.random_integers(n_rank, size=n_sample)
    w = np.random.randn(n_dim)
    X = [sigma * np.random.randn(n_dim) + w * y for y in ys]
    X = np.array(X).astype(np.float32)
    ys = np.reshape(np.array(ys), (-1, 1))
    return X, ys


def ndcg(y_true, y_score, k=10):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    y_true_sorted = sorted(y_true, reverse=True)
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
    dcg = 0
    argsort_indices = np.argsort(y_score)[::-1]
    for i in range(k):
        dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
    ndcg = dcg / ideal_dcg
    return ndcg

def clean_datasets(dataset_filepath):
    # Read training data
    with open(dataset_filepath, 'r') as trainfile:
        train_x, train_y, train_qids, _ = pyltr.data.letor.read_dataset(trainfile)
        df_train_features = pd.DataFrame(train_x, columns=['feature_{}'.format(id) for id in range(1, train_x.shape[1]+1)])
        df_train_rel = pd.DataFrame({'rel': train_y})
        df_train_qid = pd.DataFrame({'qid': train_qids})
        df_train = pd.concat([df_train_rel, df_train_qid, df_train_features], axis=1)
        features = [x for x in df_train.columns if x not in ['rel', 'qid']]
        target = ['rel']

    # scale features
    df_train[features] = StandardScaler(with_mean=0, with_std=1).fit_transform(df_train[features])
    X_train = np.array(df_train[features])
    y_train = np.array(df_train[target])

    # filter data
    labels_data_pre_list = []
    features_data_pre_list = []
    qid_data_pre_list = []
    rel_data_pre_list = []
    for index_id, query_id in enumerate(list(set(train_qids))):
        list_index_of_query_id = [list_index for list_index, query in enumerate(train_qids) if query == query_id][0:10]
        if len(list_index_of_query_id) != 10:
            continue
        qid_data_by_query = np.repeat(query_id, 10, axis=0).reshape((10, 1))
        rel_data_by_query = np.arange(1, 11).reshape((10, 1))
        labels_data_by_query = np.identity(10)
        features_data_by_query = X_train[list_index_of_query_id, :]
        labels_data_pre_list.append(labels_data_by_query)
        qid_data_pre_list.append(qid_data_by_query)
        rel_data_pre_list.append(rel_data_by_query)
        features_data_pre_list.append(features_data_by_query)
        # print features_data_by_query.shape[0], qid_data_by_query.shape[0]
    labels_data_processed = np.vstack(labels_data_pre_list)
    qid_data_processed = np.vstack(qid_data_pre_list)
    rel_data_processed = np.vstack(rel_data_pre_list)
    features_data_processed = np.vstack(features_data_pre_list)
    return labels_data_processed, qid_data_processed, rel_data_processed, features_data_processed, X_train, y_train, df_train


if __name__ == '__main__':
    # 训练集和测试集
    labels_data_processed_train, qid_data_processed_train, rel_data_processed_train, features_data_processed_train, X_train, y_train, df_train = clean_datasets(FEATURE_FILEPATH_TRAIN)
    labels_data_processed_test, qid_data_processed_test, rel_data_processed_test, features_data_processed_test, X_test, y_test, df_test = clean_datasets(FEATURE_FILEPATH_TEST)
    features_data_processed_train = features_data_processed_train.astype('float32')
    features_data_processed_test = features_data_processed_test.astype('float32')
    n_dim = features_data_processed_train.shape[1]
    query_list_of_train = np.unique(qid_data_processed_train).tolist()
    query_list_of_test = np.unique(qid_data_processed_test).tolist()
    n_rank = 10

    n_iter = 10000
    n_hidden = 20
    loss_step = 50

    model = net.RankNet(net.MLP(n_dim, n_hidden))
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    N_train = np.shape(features_data_processed_train)[0]
    train_ndcgs = []
    test_ndcgs = []
    for step in range(n_iter):
        query_id_index = np.random.randint(len(query_list_of_train))
        query_id = query_list_of_train[query_id_index]
        list_index_of_query_id = [list_index for list_index, query in enumerate(qid_data_processed_train) if query == query_id][0:10]
        i, j = np.random.randint(len(list_index_of_query_id), size=2)
        x_i = Variable(features_data_processed_train[list_index_of_query_id[i]].reshape(1, -1))
        x_j = Variable(features_data_processed_train[list_index_of_query_id[j]].reshape(1, -1))
        y_i = Variable(rel_data_processed_train[i])
        y_j = Variable(rel_data_processed_train[j])
        model.zerograds()
        loss = model(x_i, x_j, y_i, y_j)
        loss.backward()
        optimizer.update()
        if (step + 1) % loss_step == 0:
            train_ndcg = 0.0
            test_ndcg = 0.0
            for query_id in query_list_of_train:
                list_index_of_query_id = [list_index for list_index, query in enumerate(qid_data_processed_train) if
                                          query == query_id][0:10]
                train_score = model.predictor(Variable(features_data_processed_train[list_index_of_query_id]))
                train_ndcg += ndcg(rel_data_processed_train[list_index_of_query_id], train_score.data)
            train_ndcg = train_ndcg / len(query_list_of_train)
            for query_id in query_list_of_test:
                list_index_of_query_id = [list_index for list_index, query in enumerate(qid_data_processed_test) if
                                          query == query_id][0:10]
                test_score = model.predictor(Variable(features_data_processed_test[list_index_of_query_id]))
                test_ndcg += ndcg(rel_data_processed_test, test_score.data)
            test_ndcg = test_ndcg / len(query_list_of_test)
            train_ndcgs.append(train_ndcg)
            test_ndcgs.append(test_ndcg)
            print("step: {}".format(step + 1))
            print("NDCG@10 | train: {}, test: {}".format(
                train_ndcg, test_ndcg))

    sns.set_context("poster")
    plt.plot(train_ndcgs, label="Train")
    plt.plot(test_ndcgs, label="Test")
    xx = np.linspace(0, n_iter / loss_step, num=n_iter / loss_step + 1)
    labels = np.arange(loss_step, n_iter + 1, loss_step)
    plt.xticks(xx, labels, rotation=45)
    plt.legend(loc="best")
    plt.xlabel("step")
    plt.ylabel("NDCG@100")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('./ndcg_value.png')