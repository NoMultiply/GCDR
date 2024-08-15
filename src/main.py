import os
import pickle
import random
import time
from copy import deepcopy

import torch
from tqdm import tqdm

from data_loader import Dataset
from logger import OUTPUT_ROOT, get_args, logger
from models import GCDR
from utils import print_results, set_seed, valid_model



def main():
    args, parser = get_args()
    if args.dataset == 'books':
        print('!!!!!!!!!!!!!!!!!!!!!!!')
        print('Args reset for books')
        args.ri = 500
        args.test_bs = 512
        args.bs = 512
        print('!!!!!!!!!!!!!!!!!!!!!!!')
    set_seed(args.seed)
    logger.set_log_file(args, parser)
    logger.print(args)
    device = torch.device(args.device)
    args.device = device

    dataset = Dataset(args.dataset, device=device, 
                      max_history_length=args.max_history_length,
                      min_history_length=args.min_history_length)
    
    args.nmc = min(args.nmc, len(dataset.cat2id))

    # if args.reorder != 0:
    dataset.calc_user_cat_hist()

    logger.print(dataset)

    model = GCDR(dataset, device, args)
    
    logger.print(model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    bs = args.bs
    best_score, best_model, patience = 0, None, 0
    best_epoch = 0
    train_data = dataset.seq_train_data
    start_time = time.time()
    for epoch in range(1, args.ne + 1):
        random.shuffle(train_data)
        logger.print(f'[Epoch {epoch}]')
        tqdm_batch = tqdm(
            range(0, len(train_data), bs) if args.iter == 0 else range(
                0, min(len(train_data), bs * args.iter), bs),
            bar_format='{l_bar}{r_bar}', desc='[Training]')

        total_rec_loss = 0
        total_user_loss = 0
        total_diff_loss = 0
        step = 0
        for i in tqdm_batch:
            step += 1
            batch_data = train_data[i:i + bs]
            uids, histories, lengths, pos_iids = zip(*batch_data)
            
            loss_user, loss_rec, loss_diff = model.forward_bpr(uids, histories, lengths, pos_iids)
            
            total_user_loss += loss_user.item()
            total_rec_loss += loss_rec.item()
            total_diff_loss += loss_diff.item()

            loss = loss_user + loss_rec + loss_diff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.ri == 0:
                total_rec_loss /= args.ri
                tqdm_batch.write(f'[Step {step}] loss = {total_user_loss} + {total_rec_loss} + {total_diff_loss} = '
                                 f'{total_user_loss + total_rec_loss + total_diff_loss}')
                total_user_loss = 0
                total_rec_loss = 0
                total_diff_loss = 0

        tqdm_batch.close()
        # lr_scheduler.step()

        if epoch <= args.warmup:
            continue
        
        valid_scores = valid_model(model, dataset.valid_data, dataset, args, ks=[20], diversity=False)

        score = valid_scores['ndcg@20']
        if score > best_score:
            best_score, patience = score, 0
            best_epoch = epoch
            best_model = deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= args.patience:
                logger.print(
                    f'[ ][Epoch {epoch}] {valid_scores}, best = {best_score}, patience = {patience}/{args.patience}')
                logger.print('[!!! Early Stop !!!]')
                break

        logger.print(f'[{"*" if patience == 0 else " "}][Epoch {epoch}] {valid_scores}, '
                     f'best = {best_score}, patience = {patience}/{args.patience}')

    end_time = time.time()

    if best_model is not None and not args.use_final:
        model.load_state_dict(best_model)
        logger.print(f'[Epoch] Total = {epoch}, Best = {best_epoch}')

    res = valid_model(
        model, dataset.test_data, dataset, args,
        ks=[3, 5, 10, 20],
        diversity=True)
 
    print_results(args, res, start_time, end_time)
    logger.print(f'[Epoch] Total = {epoch}, Best = {best_epoch}, Best Score = {best_score}')


if __name__ == '__main__':
    main()
