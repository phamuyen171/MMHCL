from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse

from utility.parser import parse_args
from Models import MMHCL
from utility.batch_test import *
from utility.logging import Logger
import pathlib

args = parse_args()

path_name = (
    f"uu_ii={args.User_layers}_{args.Item_layers}_{args.user_loss_ratio}_{args.item_loss_ratio}"
    f"_topk={args.topk}_t={args.temperature}_regs={args.regs}_dim={args.embed_size}_{args.ablation_target}"
)
path        = f"../{args.dataset}/{path_name}/"
record_path = f"../{args.dataset}/MM/"
pathlib.Path(path).mkdir(parents=True, exist_ok=True)
pathlib.Path(record_path).mkdir(parents=True, exist_ok=True)

# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer(object):
    def __init__(self, data_config):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.logger = Logger(path, is_debug=args.debug, target=path_name,
                             path2=record_path, ablation_target=args.ablation_target)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.lr         = args.lr
        self.emb_dim    = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers   = len(self.weight_size)
        self.regs       = args.regs
        self.decay      = self.regs

        self.UI_mat   = data_config['UI_mat'].cuda()
        self.User_mat = data_config['User_mat'].cuda()
        self.Item_mat = data_config['Item_mat'].cuda()

        # ── raw features for SGFD ──────────────────────────────────────
        v_feat    = data_config.get('v_feat', None)
        t_feat    = data_config.get('t_feat', None)
        meta_feat = data_config.get('meta_feat', None)

        if v_feat is not None:
            v_feat = v_feat.cuda()
        if t_feat is not None:
            t_feat = t_feat.cuda()
        if meta_feat is not None:
            meta_feat = meta_feat.cuda()

        self.model = MMHCL(
            self.n_users, self.n_items, self.emb_dim,
            v_feat=v_feat, t_feat=t_feat, meta_feat=meta_feat,
        ).cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

        sgfd_status = "ENABLED" if self.model.sgfd_enabled else "DISABLED"
        self.logger.logging(f"[SGFD] status={sgfd_status}  "
                            f"ce_weight={args.ce_weight}  "
                            f"kd_weight={args.kd_weight}  "
                            f"t_decay={args.t_decay}")

    def set_lr_scheduler(self):
        fac       = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, ii, uu = self.model(
                self.UI_mat, self.Item_mat, self.User_mat)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        cur_best_pre_0 = 0.
        best_recall = 0
        best_ndcg   = 0
        test_ret    = ""

        n_batch = data_generator.n_train // args.batch_size + 1

        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            contrastive_loss = 0.
            sgfd_loss_total  = 0.
            n_batch = data_generator.n_train // args.batch_size + 1

            for idx in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()

                users, pos_items, neg_items = data_generator.sample()

                ua_embeddings, ia_embeddings, ii, uu = self.model(
                    self.UI_mat, self.Item_mat, self.User_mat)

                u_g_embeddings     = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                # ── original MMHCL losses ──────────────────────────────
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

                batch_cl1 = self.model.batched_contrastive_loss(ia_embeddings, ii) * args.item_loss_ratio
                batch_cl2 = self.model.batched_contrastive_loss(ua_embeddings, uu) * args.user_loss_ratio
                batch_contrastive_loss = batch_cl1 + batch_cl2

                # ── SGFD distillation loss ─────────────────────────────
                pos_items_tensor = torch.tensor(pos_items, dtype=torch.long,
                                                device=ua_embeddings.device)
                batch_sgfd_loss = self.model.sgfd_loss(pos_items_tensor)

                # ── total loss ─────────────────────────────────────────
                batch_loss = (batch_mf_loss + batch_emb_loss + batch_reg_loss
                              + batch_contrastive_loss + batch_sgfd_loss)

                batch_loss.backward(retain_graph=False)
                self.optimizer.step()

                loss             += float(batch_loss)
                mf_loss          += float(batch_mf_loss)
                emb_loss         += float(batch_emb_loss)
                reg_loss         += float(batch_reg_loss)
                contrastive_loss += float(batch_contrastive_loss)
                sgfd_loss_total  += float(batch_sgfd_loss)

            self.lr_scheduler.step()

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings, ii, uu

            if math.isnan(loss):
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = (
                    'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f(sgfd)]'
                    % (epoch, time() - t1, loss, mf_loss, emb_loss,
                       contrastive_loss, sgfd_loss_total)
                )
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val  = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()
            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            if args.verbose > 0:
                perf_str = (
                    'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f(sgfd)], '
                    'recall=[%.5f, %.5f], precision=[%.5f, %.5f], '
                    'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]'
                    % (epoch, t2 - t1, t3 - t2,
                       loss, mf_loss, emb_loss, contrastive_loss, sgfd_loss_total,
                       ret['recall'][0], ret['recall'][-1],
                       ret['precision'][0], ret['precision'][-1],
                       ret['hit_ratio'][0], ret['hit_ratio'][-1],
                       ret['ndcg'][0], ret['ndcg'][-1])
                )
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall or ret['ndcg'][1] > best_ndcg:
                if ret['recall'][1] > best_recall:
                    best_recall = ret['recall'][1]
                if ret['ndcg'][1] > best_ndcg:
                    best_ndcg = ret['ndcg'][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging(
                    "Test_Recall@%d: %.8f   Test_Precision@%d: %.8f   Test_NDCG@%d: %.8f"
                    % (eval(args.Ks)[1], test_ret['recall'][1],
                       eval(args.Ks)[1], test_ret['precision'][1],
                       eval(args.Ks)[1], test_ret['ndcg'][1])
                )
                stopping_step = 0

            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                fname = f'Model.epoch={epoch}.pth'
                torch.save(self.model.state_dict(), os.path.join(path, fname))
                break

        self.logger.logging(str(test_ret))
        self.logger.logging_sum(f"{path_name}:{str(test_ret)}")

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores  = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores  = torch.sum(torch.mul(users, neg_items), dim=1)
        regularizer = (1. / 2 * (users ** 2).sum()
                       + 1. / 2 * (pos_items ** 2).sum()
                       + 1. / 2 * (neg_items ** 2).sum()) / self.batch_size
        mf_loss  = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss, 0.0

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices   = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values    = torch.from_numpy(sparse_mx.data)
        shape     = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


# ─── entry point ─────────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    UI_mat    = data_generator.get_UI_mat()
    User_mat  = data_generator.get_U2U_mat()

    if args.dataset == "Tiktok":
        Item_mat = data_generator.get_tiktok_I2I_Hypergraph_mul_mat()
    elif args.dataset in ["Clothing", "Sports"]:
        Item_mat = data_generator.get_I2I_Hypergraph_mul_mat()

    config['UI_mat']   = UI_mat
    config['User_mat'] = User_mat
    config['Item_mat'] = Item_mat

    # ── SGFD: load raw features & meta labels ────────────────────────────
    v_feat, t_feat, meta_feat = data_generator._load_raw_features()
    config['v_feat']     = v_feat
    config['t_feat']     = t_feat
    config['meta_feat'] = meta_feat

    if v_feat is None:
        print("[SGFD] WARNING: visual features not found – SGFD on item side will be disabled.")
    if t_feat is None:
        print("[SGFD] WARNING: textual features not found – SGFD on item side will be disabled.")   
    if meta_feat is None:
        print("[SGFD] WARNING: meta labels not found – SGFD classification loss will be non-informative.")

    trainer = Trainer(data_config=config)
    trainer.train()