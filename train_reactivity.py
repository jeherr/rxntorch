import argparse
import logging
import os

from torch.utils.data import DataLoader, random_split

from rxntorch.containers.dataset import RxnGraphDataset as RxnGD
from rxntorch.models.reactivity_network import ReactivityNet as RxnNet, ReactivityTrainer as RxnTrainer
from rxntorch.utils import collate_fn


parser = argparse.ArgumentParser()

parser.add_argument("-p", "--dataset_path", type=str, default='./data/', help="train dataset")
parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset")
parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set")
parser.add_argument("-op", "--output_path", type=str, default='./saved_models/', help="saved model path")
parser.add_argument("-o", "--output_name", required=True, type=str, help="e.g. rxntorch.model")

parser.add_argument("-b", "--batch_size", type=int, default=20, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-hs", "--hidden", type=int, default=300, help="hidden size of model layers")
parser.add_argument("-l", "--layers", type=int, default=3, help="number of layers")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
parser.add_argument("--adam_weight_decay", type=float, default=0.00, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")
parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
parser.add_argument("--cuda_devices", type=int, nargs='*', default=None, help="CUDA device ids")

parser.add_argument("--log_freq", type=int, default=50, help="printing loss every n iter: setting n")


args = parser.parse_args()

logfile = '.'.join((args.output_name, "log"))
logpath = os.path.join(args.output_path, logfile)
logging.basicConfig(level=logging.INFO, style='{', format="{asctime:s}: {message:s}",
                    datefmt="%m/%d/%Y %I:%M:%S %p", handlers=(
                    logging.FileHandler(logpath), logging.StreamHandler()))

logging.info("Loading Training Dataset {dataset} in {datapath}".format(
    dataset=args.train_dataset, datapath=args.dataset_path))
dataset = RxnGD(args.train_dataset, path=args.dataset_path)
n_samples = len(dataset)
sample = dataset[0]
afeats_size, bfeats_size, binary_size = (sample["atom_features"].shape[-1], sample["bond_features"].shape[-1],
                                        sample["binary_features"].shape[-1])

n_test, n_val = int(n_samples * 0.1), int(n_samples * 0.1)
n_train = n_samples - n_test - n_val
train_set, test_set, val_set = random_split(dataset, (n_train, n_test, n_val))

logging.info("Creating Dataloaders")
train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
test_dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

logging.info("Building Reaction scoring model")
net = RxnNet(depth=args.layers, afeats_size=afeats_size, bfeats_size=bfeats_size,
             hidden_size=args.hidden, binary_size=binary_size)

logging.info("Creating Trainer")
trainer = RxnTrainer(net, train_dataloader, test_dataloader, lr=args.lr,
                     betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                     with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

logging.info("Training Start")
for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(epoch, args.output_path)
    trainer.test(epoch)
