import argparse

from torch.utils.data import DataLoader

from rxntorch.containers.dataset import RxnGraphDataset as RxnGD
from rxntorch.models.reactivity_network import ReactivityNet as RxnNet, ReactivityTrainer as RxnTrainer


parser = argparse.ArgumentParser()

parser.add_argument("-p", "--dataset_path", type=str, default='./data/', help="train dataset")
parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset")
parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set")
parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

parser.add_argument("-hs", "--hidden", type=int, default=100, help="hidden size of model layers")
parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")

parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")

parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

args = parser.parse_args()

print("Loading Train Dataset", args.train_dataset)
train_dataset = RxnGD(args.train_dataset, path=args.dataset_path)
sample = train_dataset[0]
afeats_size, bfeats_size, binary_size = sample["atom_features"].shape[-1], sample["bond_features"].shape[-1], \
                                        sample["binary_features"].shape[-1]

print("Loading Test Dataset", args.test_dataset)
test_dataset = RxnGD(args.test_dataset, path=args.dataset_path)

#print("Creating Dataloader")
#train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
#test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
#    if test_dataset is not None else None

print("Building Reaction scoring model")
net = RxnNet(depth=args.layers, afeats_size=afeats_size, bfeats_size=bfeats_size,
             hidden_size=args.hidden, binary_size=binary_size)

print("Creating Trainer")
trainer = RxnTrainer(net, train_dataset, test_dataset, lr=args.lr,
                     betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                     with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

print("Training Start")
for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(epoch, args.output_path)

    if test_data_loader is not None:
        trainer.test(epoch)
