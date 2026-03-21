import json
import random as rd
from time import time

import numpy as np
import scipy.sparse as sp
import torch

from utility.parser import parse_args

args = parse_args()


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path + '/%d-core' % args.core
        self.batch_size = batch_size

        train_file = path + '/%d-core/train.json' % args.core
        val_file = path + '/%d-core/val.json' % args.core
        test_file = path + '/%d-core/test.json' % args.core

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test, self.n_val = 0, 0, 0
        self.neg_pools = {}
        self.exist_users = []

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))

        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except Exception:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except Exception:
                continue

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}

        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for i in train_items:
                self.R[uid, i] = 1.0
            self.train_items[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except Exception:
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except Exception:
                continue

        self.R = self.R.tocsr()
        self.R_tfidf = self.build_tfidf_R()

    def is_tiktok(self):
        return str(args.dataset).lower() == "tiktok"

    def build_tfidf_R(self):
        R = self.R.tocsr()
        df = np.array(R.sum(axis=0)).reshape(-1)
        n_users = R.shape[0]

        idf = np.log(n_users / (df + 1))

        coo = R.tocoo()
        rows, cols = coo.row, coo.col
        data = idf[cols]

        R_tfidf = sp.coo_matrix((data, (rows, cols)), shape=R.shape)
        return R_tfidf.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print(
            'n_train=%d, n_test=%d, sparsity=%.5f'
            % (
                self.n_train,
                self.n_test,
                (self.n_train + self.n_test) / (self.n_users * self.n_items),
            )
        )

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    # --------------------- Original adjacency for UI graph ---------------------
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        )
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    # --------------------- Dense normalization helpers ---------------------
    def norm_dense(self, adj, normalization='origin'):
        if normalization == 'sym':
            rowsum = torch.sum(adj, -1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
            L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        elif normalization == "2sym":
            rowsum = torch.sum(adj, -1)
            d_row_inv_sqrt = torch.pow(rowsum, -0.5)
            d_row_inv_sqrt[torch.isinf(d_row_inv_sqrt)] = 0.0
            d_row_mat_inv_sqrt = torch.diagflat(d_row_inv_sqrt)

            colsum = torch.sum(adj, -2)
            d_col_inv_sqrt = torch.pow(colsum, -0.5)
            d_col_inv_sqrt[torch.isinf(d_col_inv_sqrt)] = 0.0
            d_col_mat_inv_sqrt = torch.diagflat(d_col_inv_sqrt)

            L_norm = torch.mm(torch.mm(d_row_mat_inv_sqrt, adj), d_col_mat_inv_sqrt)

        elif normalization == 'rw':
            rowsum = torch.sum(adj, -1)
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.0
            d_mat_inv = torch.diagflat(d_inv)
            L_norm = torch.mm(d_mat_inv, adj)

        elif normalization == 'origin':
            L_norm = adj

        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        return L_norm

    # --------------------- UI / U2U matrices ---------------------
    def get_UI_mat(self, norm_type='sym'):
        print("Loading UI_mat:(" + norm_type + ")")
        t = time()
        try:
            UI_mat = torch.load(self.path + '/UI_mat_' + norm_type + ".pth")
        except Exception:
            adj_mat = sp.dok_matrix(
                (self.n_users + self.n_items, self.n_users + self.n_items),
                dtype=np.float32,
            )
            adj_mat = adj_mat.tolil()
            R = self.R.tolil()

            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todense()

            UI_mat = torch.from_numpy(adj_mat).float()
            UI_mat = self.norm_dense(UI_mat, norm_type)
            UI_mat = UI_mat.to_sparse()
            torch.save(UI_mat, self.path + '/UI_mat_' + norm_type + ".pth")

        print("End Load UI_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_UI_single_mat(self, norm_type='2sym'):
        print("Loading UI_single_mat:(" + norm_type + ")")
        t = time()
        try:
            UI_mat = torch.load(self.path + '/UI_single_mat_' + norm_type + ".pth")
        except Exception:
            adj_mat = self.R.todense()
            UI_mat = torch.from_numpy(adj_mat).float()
            UI_mat = self.norm_dense(UI_mat, norm_type)
            UI_mat = UI_mat.to_sparse()
            torch.save(UI_mat, self.path + '/UI_single_mat_' + norm_type + ".pth")
        print("End Load UI_single_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_U2U_mat(self, norm_type='rw'):
        print("Loading User_mat:(" + norm_type + ")")
        t = time()
        try:
            User_mat = torch.load(self.path + '/User_mat_' + norm_type + ".pth")
        except Exception:
            R = torch.from_numpy(self.R_tfidf.todense()).float()
            User_mat = R @ R.T
            n_user = User_mat.size(0)
            mask = torch.eye(n_user, device=User_mat.device)
            User_mat[mask > 0] = 0
            User_mat = self.norm_dense(User_mat, norm_type)
            User_mat = User_mat.to_sparse()
            torch.save(User_mat, self.path + '/User_mat_' + norm_type + ".pth")
        print("End Load User_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return User_mat

    # --------------------- Feature loading ---------------------
    def load_multimodal_features(self):
        image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))
        text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))

        image_feats = torch.tensor(image_feats).float()
        text_feats = torch.tensor(text_feats).float()

        if self.is_tiktok():
            audio_feats = np.load('../data/{}/audio_feat.npy'.format(args.dataset))
            audio_feats = torch.tensor(audio_feats).float()
            return image_feats, text_feats, audio_feats

        return image_feats, text_feats

    # --------------------- Cache path helpers ---------------------
    def get_single_modality_cache_paths(self, norm_type):
        image_cache = self.path + f'/Image_mat_{norm_type}_{args.i2i_graph_mode}_topk_{args.topk}_topm_{args.densify_topm}.pth'
        text_cache = self.path + f'/Text_mat_{norm_type}_{args.i2i_graph_mode}_topk_{args.topk}_topm_{args.densify_topm}.pth'
        audio_cache = self.path + f'/Audio_mat_{norm_type}_{args.i2i_graph_mode}_topk_{args.topk}_topm_{args.densify_topm}.pth'
        return image_cache, text_cache, audio_cache

    def get_hypergraph_cache_path(self, norm_type):
        return f"{self.path}/hypergraph_mat_{norm_type}_{args.i2i_graph_mode}_topk_{args.topk}_topm_{args.densify_topm}.pth"

    def get_hypergraph_mul_cache_path(self, norm_type):
        return f"{self.path}/hypergraph_mat_mul_{norm_type}_{args.i2i_graph_mode}_topk_{args.topk}_topm_{args.densify_topm}.pth"

    # --------------------- I2I single-modality matrices ---------------------
    def get_I2I_single_mat(self, norm_type="sym"):
        print(f"Loading I2I multi-media Hypergraph mat:({norm_type})_{args.i2i_graph_mode}_topk:{args.topk}_topm:{args.densify_topm}")
        t = time()

        image_cache, text_cache, audio_cache = self.get_single_modality_cache_paths(norm_type)

        try:
            image_adj = torch.load(image_cache)
            text_adj = torch.load(text_cache)
            if self.is_tiktok():
                audio_adj = torch.load(audio_cache)

        except Exception:
            feats = self.load_multimodal_features()

            if self.is_tiktok():
                image_feats, text_feats, audio_feats = feats
            else:
                image_feats, text_feats = feats

            image_adj = self.build_sim(image_feats)
            image_adj = self.build_i2i_graph(
                image_adj,
                topk=args.topk,
                graph_mode=args.i2i_graph_mode,
                keep_self=False
            )

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_i2i_graph(
                text_adj,
                topk=args.topk,
                graph_mode=args.i2i_graph_mode,
                keep_self=False
            )

            if self.is_tiktok():
                audio_adj = self.build_sim(audio_feats)
                audio_adj = self.build_i2i_graph(
                    audio_adj,
                    topk=args.topk,
                    graph_mode=args.i2i_graph_mode,
                    keep_self=False
                )

            image_adj = self.norm_dense(image_adj, norm_type).to_sparse()
            text_adj = self.norm_dense(text_adj, norm_type).to_sparse()

            torch.save(image_adj, image_cache)
            torch.save(text_adj, text_cache)

            if self.is_tiktok():
                audio_adj = self.norm_dense(audio_adj, norm_type).to_sparse()
                torch.save(audio_adj, audio_cache)

        print("End Load I2I media-specific mat:[%.1fs](" % (time() - t) + norm_type + ")")
        if self.is_tiktok():
            return image_adj, text_adj, audio_adj
        return image_adj, text_adj, ""

    # --------------------- I2I hypergraph matrices ---------------------
    def get_I2I_Hypergrah_mat(self, norm_type="origin"):
        print(f"Loading I2I multi-media Hypergraph mat:({norm_type})_{args.i2i_graph_mode}_topk:{args.topk}_topm:{args.densify_topm}")
        t = time()
        cache_path = self.get_hypergraph_cache_path(norm_type)

        try:
            Hypergraph = torch.load(cache_path)

        except Exception:
            image_feats, text_feats = self.load_multimodal_features()

            image_adj = self.build_sim(image_feats)
            image_adj = self.build_i2i_graph(
                image_adj,
                topk=args.topk,
                graph_mode=args.i2i_graph_mode,
                keep_self=False
            )

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_i2i_graph(
                text_adj,
                topk=args.topk,
                graph_mode=args.i2i_graph_mode,
                keep_self=False
            )

            Hypergraph = torch.cat((image_adj, text_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, cache_path)

        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_I2I_Hypergraph_mul_mat(self, norm_type="sym"):
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_{args.i2i_graph_mode}_topk:{args.topk}_topm:{args.densify_topm}")
        t = time()
        cache_path = self.get_hypergraph_mul_cache_path(norm_type)

        try:
            Hypergraph_mul = torch.load(cache_path)

        except Exception:
            Hypergraph = self.get_I2I_Hypergrah_mat("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, cache_path)

        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    # --------------------- PT versions ---------------------
    def get_I2I_Hypergrah_mat_pt(self, norm_type="origin"):
        print(f"Loading I2I multi-media Hypergraph mat PT:({norm_type})_{args.i2i_graph_mode}_topk:{args.topk}_topm:{args.densify_topm}")
        t = time()
        cache_path = self.get_hypergraph_cache_path(norm_type)

        try:
            Hypergraph = torch.load(cache_path)

        except Exception:
            image_feats = torch.load("../data/{}/img_feat.pt".format(args.dataset))
            text_feats = torch.load("../data/{}/text_feat.pt".format(args.dataset))

            image_adj = self.build_sim(image_feats)
            image_adj = self.build_i2i_graph(
                image_adj,
                topk=args.topk,
                graph_mode=args.i2i_graph_mode,
                keep_self=False
            )

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_i2i_graph(
                text_adj,
                topk=args.topk,
                graph_mode=args.i2i_graph_mode,
                keep_self=False
            )

            Hypergraph = torch.cat((image_adj, text_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, cache_path)

        print("End Load I2I multi-media Hypergraph mat PT:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_I2I_Hypergraph_mul_mat_pt(self, norm_type="sym"):
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T PT:({norm_type})_{args.i2i_graph_mode}_topk:{args.topk}_topm:{args.densify_topm}")
        t = time()
        cache_path = self.get_hypergraph_mul_cache_path(norm_type)

        try:
            Hypergraph_mul = torch.load(cache_path)

        except Exception:
            Hypergraph = self.get_I2I_Hypergrah_mat_pt("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, cache_path)

        print("End Load I2I multi-media Hypergraph mul mat*mat.T PT:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    # --------------------- Tiktok-specific versions ---------------------
    def get_tiktok_I2I_Hypergrah_mat(self, norm_type="origin"):
        print(f"Loading I2I multi-media Hypergraph mat:({norm_type})_{args.i2i_graph_mode}_topk:{args.topk}_topm:{args.densify_topm}")
        t = time()
        cache_path = self.get_hypergraph_cache_path(norm_type)

        try:
            Hypergraph = torch.load(cache_path)

        except Exception:
            image_feats, text_feats, audio_feats = self.load_multimodal_features()

            image_adj = self.build_sim(image_feats)
            image_adj = self.build_i2i_graph(
                image_adj,
                topk=args.topk,
                graph_mode=args.i2i_graph_mode,
                keep_self=False
            )

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_i2i_graph(
                text_adj,
                topk=args.topk,
                graph_mode=args.i2i_graph_mode,
                keep_self=False
            )

            audio_adj = self.build_sim(audio_feats)
            audio_adj = self.build_i2i_graph(
                audio_adj,
                topk=args.topk,
                graph_mode=args.i2i_graph_mode,
                keep_self=False
            )

            Hypergraph = torch.cat((torch.cat((image_adj, text_adj), dim=1), audio_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, cache_path)

        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_tiktok_I2I_Hypergraph_mul_mat(self, norm_type="sym"):
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_{args.i2i_graph_mode}_topk:{args.topk}_topm:{args.densify_topm}")
        t = time()
        cache_path = self.get_hypergraph_mul_cache_path(norm_type)

        try:
            Hypergraph_mul = torch.load(cache_path)

        except Exception:
            Hypergraph = self.get_tiktok_I2I_Hypergrah_mat("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, cache_path)

        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    # --------------------- Similarity / graph construction ---------------------
    def build_sim(self, context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        context_norm[torch.isnan(context_norm)] = 0
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    def build_sim_feature_nan(self, context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        context_norm[context_norm.isnan()] = 0
        sim = torch.mm(context, context.transpose(1, 0))
        return sim

    def build_knn_normalized_graph(self, adj, topk):
        """
        Legacy function kept for compatibility.
        Prefer build_i2i_graph(...).
        """
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        adj = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        adj[adj > 0] = 1.0
        return adj

    def build_knn_graph(self, adj, topk, keep_self=False):
        sim = adj.clone()

        if not keep_self:
            n = sim.size(0)
            idx = torch.arange(n, device=sim.device)
            sim[idx, idx] = -1e9

        _, knn_ind = torch.topk(sim, topk, dim=-1)
        knn_adj = torch.zeros_like(sim)
        knn_adj.scatter_(-1, knn_ind, 1.0)
        return knn_adj

    def build_mutual_knn_graph(self, adj, topk, keep_self=False):
        knn_adj = self.build_knn_graph(adj, topk, keep_self=keep_self)
        mutual_adj = knn_adj * knn_adj.T
        return mutual_adj

    def densify_graph_2hop_common_neighbors(self, base_adj, topm):
        """
        base_adj: dense binary adjacency matrix, shape [N, N]
        topm: number of 2-hop neighbors to add for each node
        """
        if topm <= 0:
            return base_adj

        # 2-hop counts: number of paths of length 2
        two_hop_scores = torch.mm(base_adj, base_adj)

        # Remove self-loops
        n = two_hop_scores.size(0)
        idx = torch.arange(n, device=two_hop_scores.device)
        two_hop_scores[idx, idx] = 0

        # Do not re-add existing 1-hop edges
        two_hop_scores[base_adj > 0] = 0

        # For each node, select top-M 2-hop candidates
        add_adj = torch.zeros_like(base_adj)

        if topm > 0:
            topm = min(topm, two_hop_scores.size(1))
            vals, inds = torch.topk(two_hop_scores, k=topm, dim=-1)

            # only keep candidates with score > 0
            valid_mask = vals > 0
            add_adj.scatter_(-1, inds, valid_mask.float())

        densified_adj = base_adj + add_adj
        densified_adj[densified_adj > 0] = 1.0
        return densified_adj
    
    def build_mutual_knn_2hop_graph(self, adj, topk, topm=2, keep_self=False):
        mutual_adj = self.build_mutual_knn_graph(adj, topk, keep_self=keep_self)
        densified_adj = self.densify_graph_2hop_common_neighbors(mutual_adj, topm=topm)
        return densified_adj

    def build_i2i_graph(self, adj, topk, graph_mode="knn", keep_self=False):
        if graph_mode == "knn":
            return self.build_knn_graph(adj, topk, keep_self=keep_self)
        elif graph_mode == "mutual_knn":
            return self.build_mutual_knn_graph(adj, topk, keep_self=keep_self)
        elif graph_mode == "mutual_knn_2hop":
            return self.build_mutual_knn_2hop_graph(adj, topk, topm=args.densify_topm, keep_self=keep_self)
        else:
            raise ValueError(f"Unsupported graph_mode: {graph_mode}")