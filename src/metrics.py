# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             mean_squared_error, roc_auc_score)

# import random
from logger import logger


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def recall_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: recall score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    n = 0
    for idx in argsort:
        if idx in ground_truth:
            n += 1
    return n / len(ground_truth)


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_accuracy_metric_old(labels, preds, metrics):
    """Calculate metrics.

    Available options are: `auc`, `rmse`, `logloss`, `acc` (accurary), `f1`, `mean_mrr`,
    `ndcg` (format like: ndcg@2;4;6;8), `hit` (format like: hit@2;4;6;8), `group_auc`.

    Args:
        labels (array-like): Labels.
        preds (array-like): Predictions.
        metrics (list): List of metric names.

    Return:
        dict: Metrics.

    Examples:
        >>> cal_accuracy_metric(labels, preds, ["ndcg@2;4;6", "group_auc"])
        {'ndcg@2': 0.4026, 'ndcg@4': 0.4953, 'ndcg@6': 0.5346, 'group_auc': 0.8096}

    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = auc
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(rmse)
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = logloss
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = acc
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = f1
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = mean_mrr
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = ndcg_temp
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = hit_temp
        elif metric.startswith('recall'):
            recall_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                recall_list = [int(token) for token in ks[1].split(';')]
            for k in recall_list:
                recall_temp = np.mean([
                    recall_score(each_labels, each_preds, k)
                    for each_labels, each_preds in zip(labels, preds)
                ])
                res['recall@{0}'.format(k)] = recall_temp
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = group_auc
        else:
            raise ValueError("Metric {0} not defined".format(metric))
    return res


def cal_accuracy_metric(labels, preds, ks):
    device = labels.device
    y_true = labels
    y_score = preds
    max_k = min(y_true.shape[-1], max(ks))
    order = torch.topk(y_score, max_k)[1]
    order_best = torch.topk(y_true, max_k)[1]
    ground_truth = torch.where(y_true == 1)[0]

    # dcg best
    discounts = torch.log2(torch.arange(max_k).to(device) + 2)
    y_take_best = torch.index_select(y_true, 0, order_best)
    gains_best = 2 ** y_take_best - 1
    best_list = gains_best / discounts

    # dcg actual
    y_take_actual = torch.index_select(y_true, 0, order)
    gains_actual = 2 ** y_take_actual - 1
    actual_list = gains_actual / discounts

    res = {}
    for k in ks:
        k = min(y_true.shape[-1], k)
        # ndcg
        best = torch.sum(best_list[:k])
        actual = torch.sum(actual_list[:k])
        ndcg = actual / best
        res["ndcg@{0}".format(k)] = ndcg.item()
        
        # hit & recall
        vs, cs = torch.cat([order[:k], ground_truth]).unique(return_counts=True)
        n = (vs[cs > 1]).size(0)
        
        recall = n / len(ground_truth)
        res["hit@{0}".format(k)] = float(n > 0)
        res['recall@{0}'.format(k)] = recall
    return res, order


def calc_single_diversity_metric(candidates, order, ks, dataset, uid):
    res = {}
    candidates = candidates.tolist()
    order = order.tolist()
    for _k in ks:
        k = min(_k, len(order))
        category_dict = {}
        # for item in candidates[order[:k]]:
        for idx in order[:k]:
            item = candidates[idx]
            for cat in dataset.item_categories[item]:
                category_dict[cat] = category_dict.get(cat, 0) + 1
        res[f'gc@{_k}'] = len(
            set(category_dict.keys()) & dataset.user_categories[uid]
        ) / len(dataset.user_categories[uid])

        n_categories = sum(category_dict.values())
        si = 1
        for v in category_dict.values():
            si -= (v / n_categories) ** 2

        res[f'si@{_k}'] = si

        category_similarities = []
        ild_similarities = []
        for i in range(k):
            for j in range(i + 1, k):
                try:
                    cat_i = dataset.item_categories[candidates[order[i]]]
                    cat_j = dataset.item_categories[candidates[order[j]]]
                except IndexError as e:
                    logger.print(candidates)
                    logger.print(order)
                    logger.print(i, j)
                    raise e
                category_similarities.append(
                    len(cat_i.intersection(cat_j)) / len(cat_i.union(cat_j)))
                ild_similarities.append(len(cat_i ^ cat_j))

        res[f'cat@{_k}'] = sum(category_similarities) / \
            len(category_similarities)
        res[f'ild@{_k}'] = sum(ild_similarities) / len(ild_similarities)

        # res[f'nov_pos@{k}'] = - 1 / np.log2(dataset.n_users) * nov_pos / k
        # res[f'nov_neg@{k}'] = - 1 / np.log2(dataset.n_users) * nov_neg / k
        # res[f'nov@{k}'] = - 1 / np.log2(dataset.n_users) * nov / k

        # res[f'pru_pos@{k}'] = -spearmanr(pos_degs, range(k))[0]
        # res[f'pru_neg@{k}'] = -spearmanr(neg_degs, range(k))[0]
        # res[f'pru@{k}'] = -spearmanr(degs, range(k))[0]

        # candidates_features = dataset.item_features[candidates[order[:k]]]
        # res[f'var@{k}'] = np.var(candidates_features, axis=0).mean()
    return res


def calc_diversity_metric(diversity_metrics):
    res = {}
    for metrics in diversity_metrics:
        for metric in metrics:
            res[metric] = res.get(metric, 0) + metrics[metric]
    for metric in res:
        res[metric] = round(res[metric] / len(diversity_metrics), 4)
    return res


def rf1(acc, div):
    return round(2 * acc * div / (acc + div), 4)


def main():
    labels = np.array([[1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0]])
    # preds = np.array([[0.5331, 0.9007, 0.7041, 0.3999, 0.3578, 0.023, 0.7928, 0.3715, 0.427, 0.2602, 0.65]])
    # logger.print(cal_accuracy_metric(labels, preds, ['ndcg@1;3;5;10;20', 'mean_mrr', 'recall@1;3;5;10;20']))
    rec = [0, 6, 8, 4, 3, 5, 7, 10, 9, 1, 2]
    # random.shuffle(rec)
    preds_1 = len(rec) - np.argsort(rec)

    rec_top_7 = rec[:7]
    scores = list(range(len(rec_top_7), 0, -1))
    preds_2 = np.zeros(len(rec))
    preds_2[rec_top_7] = scores

    scores = np.array(range(len(rec), 0, -1))
    preds_3 = np.zeros(len(rec))
    preds_3[rec] = scores
    # logger.print(rec)
    # logger.print(preds_3)
    # logger.print(np.argsort(preds_3)[::-1])
    # logger.print(cal_accuracy_metric(labels, preds_1[None, :], [
    #              'ndcg@1;3;5;7', 'mean_mrr', 'recall@1;3;5;7']))
    # logger.print(cal_accuracy_metric(labels, preds_2[None, :], [
    #              'ndcg@1;3;5;7', 'mean_mrr', 'recall@1;3;5;7']))
    # logger.print(cal_accuracy_metric(labels, preds_3[None, :], [
    #              'ndcg@1;3;5;7', 'mean_mrr', 'recall@1;3;5;7']))

    preds = np.random.random(labels.shape)
    logger.print(sorted(cal_accuracy_metric_old(labels, preds, ['ndcg@1;3;5;7', 'hit@1;3;5;7', 'recall@1;3;5;7']).items()))
    logger.print(sorted(cal_accuracy_metric(torch.tensor(labels[0]), torch.tensor(preds[0]), [1, 3, 5, 7])[0].items()))


if __name__ == '__main__':
    main()
