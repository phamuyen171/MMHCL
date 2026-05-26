# import os
# import numpy as np
# from time import time

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import copy

# from utility.parser import parse_args
# from utility.norm import build_sim, build_knn_normalized_graph
# args = parse_args()

# class LightGCN(nn.Module):
#     def __init__(self, n_users, n_items, embedding_dim):
#         super().__init__()
#         self.n_users = n_users
#         self.n_items = n_items
#         self.embedding_dim = embedding_dim


#         self.user_embedding = nn.Embedding(n_users, embedding_dim)
#         self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
#         nn.init.xavier_uniform_(self.user_embedding.weight)
#         nn.init.xavier_uniform_(self.item_id_embedding.weight)

#     def forward(self, adj):
#         ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
#         all_embeddings = [ego_embeddings]
#         for i in range(args.UI_layers):
#             side_embeddings = torch.sparse.mm(adj, ego_embeddings)
#             ego_embeddings = side_embeddings
#             all_embeddings += [ego_embeddings]
#         all_embeddings = torch.stack(all_embeddings, dim=1)
#         all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
#         u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
#         return u_g_embeddings, i_g_embeddings



# class MMHCL(nn.Module):
#     def __init__(self, n_users, n_items, embedding_dim):
#         super(MMHCL, self).__init__()
#         self.n_users = n_users
#         self.n_items = n_items
#         self.embeddings_dim = embedding_dim

#         self.user_ui_embedding = nn.Embedding(n_users, self.embeddings_dim)
#         self.item_ui_embedding = nn.Embedding(n_items, self.embeddings_dim)

#         self.uu_embedding = nn.Embedding(n_users, self.embeddings_dim)
#         self.ii_embedding = nn.Embedding(n_items, self.embeddings_dim)

#         if args.cf_model == 'NGCF':
#             self.GC_Linear_list = nn.ModuleList()
#             self.Bi_Linear_list = nn.ModuleList()
#             self.dropout_list = nn.ModuleList()
#             for i in range(args.UI_layers):
#                 self.GC_Linear_list.append(nn.Linear(eval(args.weight_size)[i], eval(args.weight_size)[i + 1]))
#                 self.Bi_Linear_list.append(nn.Linear(eval(args.weight_size)[i], eval(args.weight_size)[i + 1]))
#                 self.dropout_list.append(nn.Dropout(0.1))

#         nn.init.xavier_uniform_(self.user_ui_embedding.weight)
#         nn.init.xavier_uniform_(self.item_ui_embedding.weight)
#         nn.init.xavier_uniform_(self.uu_embedding.weight)
#         nn.init.xavier_uniform_(self.ii_embedding.weight)

#         self.tau = args.temperature

#     def forward(self, UI_mat, I2I_mat, U2U_mat):

#         ii_emb = self.ii_embedding.weight
#         uu_emb = self.uu_embedding.weight

#         if args.item_loss_ratio != 0:
#             for i in range(args.Item_layers):
#                 ii_emb = torch.sparse.mm(I2I_mat, ii_emb)

#         if args.user_loss_ratio != 0:
#             for i in range(args.User_layers):
#                 uu_emb = torch.sparse.mm(U2U_mat, uu_emb)

#         if args.cf_model == 'LightGCN':
#             ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
#             all_embeddings = [ego_embeddings]
#             for i in range(args.UI_layers):
#                 side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
#                 ego_embeddings = side_embeddings
#                 all_embeddings += [ego_embeddings]
#             all_embeddings = torch.stack(all_embeddings, dim=1)
#             all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
#             u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

#         elif args.cf_model == 'NGCF':
#             ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
#             all_embeddings = [ego_embeddings]
#             for i in range(args.UI_layers):
#                 side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
#                 sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
#                 bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
#                 bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
#                 ego_embeddings = sum_embeddings + bi_embeddings
#                 ego_embeddings = self.dropout_list[i](ego_embeddings)

#                 norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
#                 all_embeddings += [norm_embeddings]

#             all_embeddings = torch.stack(all_embeddings, dim=1)
#             all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
#             u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
#         elif args.cf_model == 'MF':
#             u_ui_emb, i_ui_emb=self.user_ui_embedding.weight, self.item_ui_embedding.weight

#         if args.item_loss_ratio != 0:
#             i_ui_emb = i_ui_emb + F.normalize(ii_emb, p=2, dim=1)

#         if args.user_loss_ratio != 0:
#             u_ui_emb = u_ui_emb + F.normalize(uu_emb, p=2, dim=1)



#         return u_ui_emb, i_ui_emb, ii_emb, uu_emb

#     def batched_contrastive_loss(self, z1, z2, batch_size=4096):
#         device = z1.device
#         num_nodes = z1.size(0)
#         num_batches = (num_nodes - 1) // batch_size + 1
#         f = lambda x: torch.exp(x / self.tau)
#         indices = torch.arange(0, num_nodes).to(device)
#         losses = []

#         for i in range(num_batches):
#             mask = indices[i * batch_size:(i + 1) * batch_size]
#             refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
#             between_sim = f(self.sim(z1[mask], z2))  # [B, N]

#             losses.append(-torch.log(
#                 between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
#                 / (refl_sim.sum(1) + between_sim.sum(1)
#                    - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
#         loss_vec = torch.cat(losses)
#         return loss_vec.mean()

#     def sim(self, z1, z2):
#         z1 = F.normalize(z1)
#         z2 = F.normalize(z2)
#         return torch.mm(z1, z2.t())

import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph

# SGFD modules
from feature_extractor.FeatureExtractorModel import FeatureExtractorModel
from feature_fusion.FeatureFusionModel import FeatureFusionModel

args = parse_args()


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(args.UI_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings


class MMHCL(nn.Module):
    """
    MMHCL with SGFD feature distillation integrated.

    SGFD adds per-modality Teacher-Student feature extractors and a
    cross-modal fusion classifier.  The resulting distillation loss
    (mfd_loss) is combined with MMHCL's original BPR + contrastive loss,
    mirroring the same integration pattern used in BM3+SGFD.

    New constructor arguments (passed via args):
        --v_feat_dim   : visual feature raw dimension  (e.g. 4096)
        --t_feat_dim   : textual feature raw dimension (e.g. 384)
        --sgfd_t       : temperature for knowledge distillation (default 100)
        --ce_weight    : weight for classification losses   (default 0.1)
        --kd_weight    : weight for distillation losses     (default 0.1)
        --sgfd_weight  : overall scale of mfd_loss          (default 1.0)
        --dropout      : dropout rate for target embeddings (default 0.3)
    """

    def __init__(self, n_users, n_items, embedding_dim,
                 v_feat=None, t_feat=None, meta_label=None):
        super(MMHCL, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embeddings_dim = embedding_dim

        # ── original MMHCL embeddings ────────────────────────────────────
        self.user_ui_embedding = nn.Embedding(n_users, self.embeddings_dim)
        self.item_ui_embedding = nn.Embedding(n_items, self.embeddings_dim)
        self.uu_embedding = nn.Embedding(n_users, self.embeddings_dim)
        self.ii_embedding = nn.Embedding(n_items, self.embeddings_dim)

        if args.cf_model == 'NGCF':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(args.UI_layers):
                self.GC_Linear_list.append(nn.Linear(eval(args.weight_size)[i], eval(args.weight_size)[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(eval(args.weight_size)[i], eval(args.weight_size)[i + 1]))
                self.dropout_list.append(nn.Dropout(0.1))

        nn.init.xavier_uniform_(self.user_ui_embedding.weight)
        nn.init.xavier_uniform_(self.item_ui_embedding.weight)
        nn.init.xavier_uniform_(self.uu_embedding.weight)
        nn.init.xavier_uniform_(self.ii_embedding.weight)

        self.tau = args.temperature

        # ── SGFD modules ─────────────────────────────────────────────────
        self.v_feat = v_feat          # [n_items, v_feat_dim] tensor or None
        self.t_feat = t_feat          # [n_items, t_feat_dim] tensor or None
        self.meta_label = meta_label  # [n_items] LongTensor of category ids

        self.sgfd_t       = getattr(args, 'sgfd_t',      100)
        self.ce_weight    = getattr(args, 'ce_weight',   0.1)
        self.kd_weight    = getattr(args, 'kd_weight',   0.1)
        self.sgfd_weight  = getattr(args, 'sgfd_weight', 1.0)
        self.dropout_rate = getattr(args, 'dropout',     0.3)

        self.v_feature_extractor = None
        self.t_feature_extractor = None
        self.feature_fusion_model = None

        if self.v_feat is not None and meta_label is not None:
            self.v_feature_extractor = FeatureExtractorModel(
                self.v_feat, self.v_feat.shape[1],
                meta_label, dim_latent=embedding_dim, t=self.sgfd_t
            )
        if self.t_feat is not None and meta_label is not None:
            self.t_feature_extractor = FeatureExtractorModel(
                self.t_feat, self.t_feat.shape[1],
                meta_label, dim_latent=embedding_dim, t=self.sgfd_t
            )
        if (self.v_feat is not None or self.t_feat is not None) and meta_label is not None:
            self.feature_fusion_model = FeatureFusionModel(meta_label, embedding_dim)

    # ─────────────────────────────────────────────────────────────────────
    # Original MMHCL forward (unchanged)
    # ─────────────────────────────────────────────────────────────────────
    def forward(self, UI_mat, I2I_mat, U2U_mat):
        ii_emb = self.ii_embedding.weight
        uu_emb = self.uu_embedding.weight

        if args.item_loss_ratio != 0:
            for i in range(args.Item_layers):
                ii_emb = torch.sparse.mm(I2I_mat, ii_emb)

        if args.user_loss_ratio != 0:
            for i in range(args.User_layers):
                uu_emb = torch.sparse.mm(U2U_mat, uu_emb)

        if args.cf_model == 'LightGCN':
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(args.UI_layers):
                side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        elif args.cf_model == 'NGCF':
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(args.UI_layers):
                side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)
                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        elif args.cf_model == 'MF':
            u_ui_emb = self.user_ui_embedding.weight
            i_ui_emb = self.item_ui_embedding.weight

        if args.item_loss_ratio != 0:
            i_ui_emb = i_ui_emb + F.normalize(ii_emb, p=2, dim=1)

        if args.user_loss_ratio != 0:
            u_ui_emb = u_ui_emb + F.normalize(uu_emb, p=2, dim=1)

        return u_ui_emb, i_ui_emb, ii_emb, uu_emb

    # ─────────────────────────────────────────────────────────────────────
    # SGFD distillation loss  (mirrors BM3's calculate_loss structure)
    # ─────────────────────────────────────────────────────────────────────
    def sgfd_loss(self, pos_items):
        """
        Compute SGFD multi-modal feature distillation loss for the batch of
        positive items.  Returns a scalar tensor (0 if no SGFD modules exist).

        Steps (identical to BM3+SGFD):
          1. For each modality, run FeatureExtractorModel → teacher_x,
             classification loss, KD loss, feature-constraint loss.
          2. Fuse student features across modalities via FeatureFusionModel.
          3. Combine: mfd_loss = ce_weight*(class+fusion) + kd_weight*(kd+feat)
        """
        if self.feature_fusion_model is None:
            return torch.tensor(0.0, device=self.user_ui_embedding.weight.device)

        items = pos_items.long()
        num_modal = 0

        t_student_feat = v_student_feat = None
        t_class_loss = t_kd_loss = t_feature_loss = 0.0
        v_class_loss = v_kd_loss = v_feature_loss = 0.0

        if self.t_feature_extractor is not None:
            t_student_feat, t_class_loss, t_kd_loss, t_feature_loss = \
                self.t_feature_extractor(items)
            num_modal += 1

        if self.v_feature_extractor is not None:
            v_student_feat, v_class_loss, v_kd_loss, v_feature_loss = \
                self.v_feature_extractor(items)
            num_modal += 1

        if num_modal == 0:
            return torch.tensor(0.0, device=self.user_ui_embedding.weight.device)

        class_loss   = (t_class_loss   + v_class_loss)   / num_modal
        kd_loss      = (t_kd_loss      + v_kd_loss)      / num_modal
        feature_loss = (t_feature_loss + v_feature_loss) / num_modal

        # Cross-modal fusion loss
        if t_student_feat is not None and v_student_feat is not None:
            fused = torch.cat([t_student_feat, v_student_feat], dim=1)
            fusion_loss = self.feature_fusion_model(items, fused, has_n=True)
        elif t_student_feat is not None:
            # single-modality fallback: self-fusion
            fused = torch.cat([t_student_feat, t_student_feat], dim=1)
            fusion_loss = self.feature_fusion_model(items, fused, has_n=True)
        else:
            fused = torch.cat([v_student_feat, v_student_feat], dim=1)
            fusion_loss = self.feature_fusion_model(items, fused, has_n=True)

        mfd_loss = (self.ce_weight * (class_loss + fusion_loss) +
                    self.kd_weight * (kd_loss + feature_loss))

        return self.sgfd_weight * mfd_loss

    # ─────────────────────────────────────────────────────────────────────
    # Contrastive helpers (unchanged from MMHCL)
    # ─────────────────────────────────────────────────────────────────────
    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim    = f(self.sim(z1[mask], z1))   # [B, N]
            between_sim = f(self.sim(z1[mask], z2))   # [B, N]
            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())
            ))
        return torch.cat(losses).mean()

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
