import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import tqdm

from .wln import WLNet
from .attention import Attention
from .reactivity_scoring import ReactivityScoring
from .utils import ScheduledOptim


class ReactivityNet(nn.Module):
    def __init__(self, depth, afeats_size, bfeats_size, hidden_size, binary_size):
        super(ReactivityNet, self).__init__()
        self.hidden_size = hidden_size
        self.wln = WLNet(depth, afeats_size, bfeats_size, hidden_size)
        self.attention = Attention(hidden_size, binary_size)
        self.reactivity_scoring = ReactivityScoring(hidden_size, binary_size)

    def forward(self, fatoms, fbonds, atom_nb, bond_nb, num_nbs, n_atoms, binary_feats, blabels):
        local_features = self.wln(fatoms, fbonds, atom_nb, bond_nb, num_nbs, n_atoms)
        local_pair, global_pair = self.attention(local_features, binary_feats)
        pair_scores = self.reactivity_scoring(local_pair, global_pair, binary_feats)
        masked_scores = torch.where((blabels == -1.0), pair_scores - 10000, pair_scores)
        _, top_k = torch.topk(torch.flatten(masked_scores, start_dim=1, end_dim=-1), 20)
        return pair_scores, top_k


class ReactivityTrainer(nn.Module):
    def __init__(self, rxn_net, train_dataloader, test_dataloader, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01,
                 warmup_steps=10000, with_cuda=True, cuda_devices=None, log_freq=10):
        super(ReactivityTrainer, self).__init__()
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.model = rxn_net
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optimizer, self.model.hidden_size, n_warmup_steps=warmup_steps)
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iterate(epoch, self.train_data)

    def test(self, epoch):
        self.iterate(epoch, self.test_data, train=False)

    def iterate(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            pair_scores, top_k = self.model.forward(data['atom_features'], data['bond_features'], data['atom_graph'],
                                                    data['bond_graph'], data['n_bonds'], data['n_atoms'],
                                                    data['binary_features'], data['bond_labels'])
            loss = F.binary_cross_entropy_with_logits(pair_scores, data['bond_labels'], reduction='none')
            loss *= torch.ne(data['bond_labels'], -1).float()
            loss = torch.sum(loss)
            avg_loss += loss.item()

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # Gather the indices of bond labels where a bond changes to calculate accuracy
            sp_labels = torch.stack(torch.where(torch.flatten(data['bond_labels'],
                                                              start_dim=1, end_dim=-1) == 1), dim=-1)
            batch_size, nk = top_k.shape[0], top_k.shape[1]
            sp_top_k = torch.empty((batch_size * nk, 2), dtype=torch.int64)
            for j in range(batch_size):
                for k in range(nk):
                    sp_top_k[j * nk + k, 0], sp_top_k[j * nk + k, 1] = j, top_k[j, k]

            sum_acc, sum_err = 0.0, 0.0
            mol_label_idx = [torch.where(sp_labels[:,0] == i)[0] for i in range(batch_size)]
            mol_topk_idx = [torch.where(sp_top_k[:,0] == i)[0] for i in range(batch_size)]
            mol_labels = [sp_labels[idx,1] for idx in mol_label_idx]
            mol_topk = [sp_top_k[idx,1] for idx in mol_topk_idx]
            hits = [(mol_labels[i].unsqueeze(0) == mol_topk[i].unsqueeze(1)).any(dim=-1) for i in range(batch_size)]
            print(hits)

            #print(torch.where((sp_labels.unsqueeze(0) == sp_top_k.unsqueeze(1)).all(dim=-1)))


            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                # "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))  # , "total_acc=",
        # total_correct * 100.0 / total_element)

    def save(self, epoch, file_path="output/trained.model"):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
