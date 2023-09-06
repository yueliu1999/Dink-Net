import os
import wandb
import argparse
import warnings
from utils import *
from tqdm import tqdm
from model import DinkNet, DinkNet_dgl
warnings.filterwarnings("ignore")

def train(args=None):

    # setup random seed
    setup_seed(args.seed)
    
    # load graph data
    if args.dataset in ["cora", "citeseer"]:
        x, adj, y, n, k, d = load_data(args)
    elif args.dataset in ["amazon_photo"]:
        x, adj, y, n, k, d = load_amazon_photo()
    elif args.dataset in ["ogbn_arxiv"]:
        x, adj, y, n, k, d, train_loader, test_loader = load_data_ogb(args)

    # model
    if args.dataset in ["cora", "citeseer"]:
        model = DinkNet(n_in=d, n_h=args.hid_units, n_cluster=k, tradeoff=args.tradeoff, activation=args.activate)
    elif args.dataset in ["amazon_photo", "ogbn_arxiv"]:
        model = DinkNet_dgl(g_global=adj, n_in=d, n_h=args.hid_units, n_cluster=k,
                            tradeoff=args.tradeoff, encoder_layers=args.encoder_layer, 
                            activation=args.activate, projector_layers=args.projector_layer, dropout_rate=args.dropout_rate)

    # to device, batch graph: x and adj on CPU
    if args.batch_train and args.batch_test:
        model = model.to(args.device)
    # to device, full graph: all in GPU
    else:
        x, adj, model = map(lambda tmp: tmp.to(args.device), [x, adj, model])

    # testing
    y_hat = model.clustering(x, adj, finetune=False)

    acc, nmi, ari, f1 = evaluation(y, y_hat)

    # logging
    tqdm.write("test      ｜ acc:{:.2f} ｜ nmi:{:.2f} ｜ ari:{:.2f} ｜ f1:{:.2f}".format(acc, nmi, ari, f1))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    # training
    if args.wandb:
        if not os.path.exists("./wandb/"):
            os.makedirs("./wandb")

        wandb.init(config=args,
                   project="ICML23_DinkNet",
                   name="baseline_{}".format(args.dataset),
                   dir="./wandb/",
                   job_type="training",
                   reinit=True)

    for epoch in tqdm(range(args.epochs)):

        model.train()
        # batch graph training
        if args.batch_train:
            for input_nodes, _, blocks in train_loader:
                optimizer.zero_grad()

                x_batch = x[blocks[0].srcdata["_ID"]]
                # to device
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(args.device)
                x_batch = x_batch.to(args.device)

                loss, sample_center_distance = model.cal_loss(x_batch, blocks, batch_train=True, finetune=False)

                loss.backward()
                optimizer.step()

        # full graph training
        else:
            optimizer.zero_grad()
            loss, sample_center_distance = model.cal_loss(x, adj, finetune=False)
            loss.backward()
            optimizer.step()

        # evaluation
        if (epoch + 1) % args.eval_inter == 0:
            model.eval()

            y_hat = model.clustering(x, adj, finetune=False)

            acc, nmi, ari, f1 = evaluation(y, y_hat)
            print(acc, nmi, ari, f1)
            if best_acc < acc:
                best_acc = acc
                torch.save(model.state_dict(), "./models/DinkNet_" + args.dataset + ".pt")

            # logging
            tqdm.write("epoch {:03d} ｜ acc:{:.2f} ｜ nmi:{:.2f} ｜ ari:{:.2f} ｜ f1:{:.2f}".format(epoch, acc, nmi, ari, f1))

            if args.wandb:
                wandb.log({"epoch": epoch, "loss": loss, "acc": acc, "nmi": nmi, "ari": ari, "f1": f1})
        else:

            if args.wandb:
                wandb.log({"epoch": epoch, "loss": loss})

    # testing
    model.load_state_dict(torch.load("./models/DinkNet_" + args.dataset + ".pt"))
    model.eval()

    y_hat = model.clustering(x, adj)

    acc, nmi, ari, f1 = evaluation(y, y_hat)

    # logging
    tqdm.write("test      ｜ acc:{:.2f} ｜ nmi:{:.2f} ｜ ari:{:.2f} ｜ f1:{:.2f}".format(acc, nmi, ari, f1))

    if args.wandb:
        wandb.log({"epoch": epoch, "loss": loss, "acc": acc, "nmi": nmi, "ari": ari, "f1": f1})


if __name__ == '__main__':

    # hyper-parameter settings
    parser = argparse.ArgumentParser("DinkNet")

    # data
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="training device")
    parser.add_argument("--dataset", type=str, default="citeseer", help="dataset name")
    parser.add_argument("--dataset_dir", type=str, default="../data", help="dataset root path")

    # model
    parser.add_argument("--tradeoff", type=float, default=1e-10, help="tradeoff parameter")
    parser.add_argument("--activate", type=str, default="prelu", help="activation function")
    parser.add_argument("--hid_units", type=int, default=1536, help="number of hidden units")
    parser.add_argument("--encoder_layer", type=int, default=1, help="number of encoder layers")
    parser.add_argument("--projector_layer", type=int, default=1, help="number of projector layers")

    # training
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wandb", action='store_true', default=False, help="enable wandb")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_train", action='store_true', default=False, help="batch train")
    parser.add_argument("--batch_test", action='store_true', default=False, help="batch test")
    parser.add_argument("--batch_size", type=int, default=-1, help="batch size")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--eval_inter", type=int, default=10, help="interval of evaluation")

    args = parser.parse_args()

    train(args=args)
