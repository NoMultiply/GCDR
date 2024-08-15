import random

import numpy as np
import torch
from logger import logger
from metrics import (cal_accuracy_metric, calc_diversity_metric,
                     calc_single_diversity_metric, rf1)
from tqdm import tqdm
from models import GCDR
from data_loader import Dataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def difference(ta, tb):
    vs, cs = torch.cat([ta, tb]).unique(return_counts=True)
    # tc = vs[cs > 1]
    # vs, cs = torch.cat([ta, tc]).unique(return_counts=True)
    return vs[cs == 1]


def intersection(ta, tb):
    vs, cs = torch.cat([ta, tb]).unique(return_counts=True)
    return vs[cs > 1]


def valid_model(model: GCDR, data, dataset: Dataset, args, ks=None, diversity=False, max_candidates=20000):
    if args.mc:
        return valid_model_mc(model, data, dataset, args, ks, diversity, max_candidates)
    return valid_model_without_mc(model, data, dataset, args, ks, diversity, max_candidates)

def valid_model_without_mc(model: GCDR, data, dataset: Dataset, args, ks=None, diversity=False, max_candidates=20000):
    if ks is None:
        ks = [3, 5, 10, 20]
    model.eval()
    all_items = torch.LongTensor(list(range(dataset.n_items))).to(args.device)
    accuracy_metrics = []
    diversity_metrics = []
    with torch.no_grad():
        user_embeddings = []
        bs = args.test_bs
        for i in tqdm(range(0, dataset.n_users, bs), bar_format='{l_bar}{r_bar}', desc='[Valid Model][User Embeddings]'):
            batch_users = list(range(i, min(i + bs, dataset.n_users)))
            batch_data = [dataset.seq_test_data[uid] for uid in batch_users]
            batch_histories, batch_lengths = zip(*batch_data)
            user_embeddings.append(model.get_user_embeddings(batch_users, batch_histories, batch_lengths)[0])

        user_embeddings = torch.concat(user_embeddings)

        item_embeddings = model.get_item_embeddings(list(range(dataset.n_items)))[0]

        for uid, pos_iids in tqdm(data.items(), bar_format='{l_bar}{r_bar}', desc='[Valid Model]'):
            candidates_tensor = all_items
            
            if dataset.n_items > max_candidates:
                idx = torch.randint(0, dataset.n_items, (max_candidates, )).to(args.device)
                candidates_tensor = candidates_tensor[idx]
                candidates_tensor = torch.cat([candidates_tensor, pos_iids]).unique()

            if uid in dataset.train_data:
                candidates_tensor = difference(candidates_tensor, dataset.train_data_tensor[uid])
           
            labels = torch.isin(candidates_tensor, pos_iids).long()
            
            preds = torch.sum(
                user_embeddings[uid].unsqueeze(0) * item_embeddings[candidates_tensor],  # (1 x d) x (n x d) -> (n x d)
                dim=-1
            )

            accuracy_metric, order = cal_accuracy_metric(labels, preds, ks)
            accuracy_metrics.append(accuracy_metric)
            if diversity:
                diversity_metrics.append(calc_single_diversity_metric(
                    candidates_tensor, order, ks, dataset, uid))
                
    accuracy_metrics = calc_diversity_metric(accuracy_metrics)
    model.train()
    if not diversity:
        return accuracy_metrics
    diversity_metrics = calc_diversity_metric(diversity_metrics)
    accuracy_metrics.update(diversity_metrics)
    return accuracy_metrics


def valid_model_mc(model: GCDR, data, dataset: Dataset, args, ks=None, diversity=False, max_candidates=20000):
    if ks is None:
        ks = [3, 5, 10, 20]
    model.eval()
    all_items = torch.LongTensor(list(range(dataset.n_items))).to(args.device)
    accuracy_metrics = []
    diversity_metrics = []
    bs = args.test_bs

    if args.mc_func == 'avg':
        mc_func = lambda x: (torch.mean(x, dim=-1), None)
    else:  # elif args.mc_func == 'max':
        mc_func = lambda x: torch.max(x, dim=-1)

    with torch.no_grad():
        all_user_embeddings = []

        user_embeddings = []
        # for i in tqdm(range(0, dataset.n_users, bs), bar_format='{l_bar}{r_bar}', desc=f'[Valid Model][User Embeddings][mc {args.mc}][nmc 0]'):
        for i in range(0, dataset.n_users, bs):
            batch_users = list(range(i, min(i + bs, dataset.n_users)))
            batch_data = [dataset.seq_test_data[uid] for uid in batch_users]
            batch_histories, batch_lengths = zip(*batch_data)
            user_embeddings.append(model.get_user_embeddings(batch_users, batch_histories, batch_lengths)[0])  # bs x d

        user_embeddings = torch.concat(user_embeddings)
        all_user_embeddings.append(user_embeddings)

        user_cat_hist_topk_indices = dataset.user_cat_hist.topk(args.nmc)[1]
        for nmc in range(args.nmc):
            user_cat_hist = torch.zeros_like(dataset.user_cat_hist)
            user_cat_hist[torch.arange(user_cat_hist.size(0)), user_cat_hist_topk_indices[:, nmc]] = 1.0
            user_embeddings = []
            # for i in tqdm(range(0, dataset.n_users, bs), bar_format='{l_bar}{r_bar}', desc=f'[Valid Model][User Embeddings][mc {args.mc}][nmc {nmc + 1}]'):
            for i in range(0, dataset.n_users, bs):
                batch_users = list(range(i, min(i + bs, dataset.n_users)))
                batch_data = [dataset.seq_test_data[uid] for uid in batch_users]
                batch_histories, batch_lengths = zip(*batch_data)
                user_embeddings.append(model.get_user_embeddings(batch_users, batch_histories, batch_lengths, user_cat_hist=user_cat_hist)[0])  # bs x d

            user_embeddings = torch.concat(user_embeddings)  # n_user x d
            all_user_embeddings.append(user_embeddings)
        all_user_embeddings = torch.stack(all_user_embeddings, dim=1)  # n_user x (nmc + 1) x d

        item_embeddings = model.get_item_embeddings(list(range(dataset.n_items)))[0]

        for uid, pos_iids in tqdm(data.items(), bar_format='{l_bar}{r_bar}', desc='[Valid Model]'):
            candidates_tensor = all_items
            
            if dataset.n_items > max_candidates:
                idx = torch.randint(0, dataset.n_items, (max_candidates, )).to(args.device)
                candidates_tensor = candidates_tensor[idx]
                candidates_tensor = torch.cat([candidates_tensor, pos_iids]).unique()

            if uid in dataset.train_data:
                candidates_tensor = difference(candidates_tensor, dataset.train_data_tensor[uid])
           
            labels = torch.isin(candidates_tensor, pos_iids).long()
            
            preds = torch.sum(
                all_user_embeddings[uid].unsqueeze(0) * item_embeddings[candidates_tensor].unsqueeze(1),  # (1 x nmc x d) x (n x 1 x d) -> (n x nmc x d)
                dim=-1
            )  # n x nmc

            preds, _ = mc_func(preds)  # n

            accuracy_metric, order = cal_accuracy_metric(labels, preds, ks)
            accuracy_metrics.append(accuracy_metric)
            if diversity:
                diversity_metrics.append(calc_single_diversity_metric(
                    candidates_tensor, order, ks, dataset, uid))
                
    accuracy_metrics = calc_diversity_metric(accuracy_metrics)
    model.train()
    if not diversity:
        return accuracy_metrics
    diversity_metrics = calc_diversity_metric(diversity_metrics)
    accuracy_metrics.update(diversity_metrics)
    return accuracy_metrics


def print_results(args, res, start_time, end_time):
    logger.print()
    logger.print(args)
    logger.print()
    logger.print(res)
    logger.print()
    logger.print('[Testing]')
    logger.print()

    acc_tag = 'ndcg'
    div_tag = 'si'

    for k in (3, 5, 10, 20):
        logger.print(f'NDCG@{k}:', res[f'ndcg@{k}'])
        logger.print(f'Hit@{k}:', res[f'hit@{k}'])
        logger.print(f'Recall@{k}:', res[f'recall@{k}'])
        logger.print(f'ILCS@{k}:', res[f'cat@{k}'])
        logger.print(f'ICSI@{k}:', res[f'si@{k}'])
        logger.print(f'ILD@{k}:', res[f'ild@{k}'])
        logger.print(f'GC@{k}:', res[f'gc@{k}'])
        logger.print(f'RF1@{k}', rf1(res[f'{acc_tag}@{k}'], res[f'{div_tag}@{k}']))
        logger.print()
    
    logger.print(f"'NDCG': {res['ndcg@20']}, 'Hit': {res['hit@20']}, "
                 f"'Recall': {res['recall@20']}, 'ILCS': {res['cat@20']}, "
                 f"'ICSI': {res['si@20']}, 'ILD': {res['ild@20']}, 'GC': {res['gc@20']}, "
                 f"'RF1': {rf1(res[acc_tag + '@20'], res[div_tag + '@20'])}")
    logger.print()

    logger.print(f'Training Time: {end_time - start_time}s')
