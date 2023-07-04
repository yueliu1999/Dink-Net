import os
import wandb
import argparse
from utils import *
from tqdm import tqdm
from model import DinkNet


def train(args=None):

    # setup random seed
    setup_seed(args.seed)
    
    # load graph data
    if args.dataset in ["cora", "citeseer"]:
        x, adj, y, n, k, d = load_data(args.dataset)

    # label of discriminative task
    disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0)

    # model
    model = DinkNet(d, args.hid_units, k, args.tradeoff, args.activate)

    # to device
    x, adj, disc_y, model = map(lambda tmp: tmp.to(args.device), [x, adj, disc_y, model])

    # load pre-trained model parameter
    model.load_state_dict(torch.load("./models/DinkNet_{}.pt".format(args.dataset)))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

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
        optimizer.zero_grad()

        # augmentations
        x_aug = aug_feature_dropout(x)
        x_shuffle = aug_feature_shuffle(x_aug)

        # loss
        loss, sample_center_distance = model.cal_loss(x, x_aug, x_shuffle, adj, disc_y)

        loss.backward()
        optimizer.step()

        # evaluation
        if (epoch + 1) % args.eval_inter == 0:
            model.eval()
            y_hat = model.clustering(x, adj)
            acc, nmi, ari, f1 = evaluation(y, y_hat)

            if best_acc < acc:
                best_acc = acc
                torch.save(model.state_dict(), "./models/DinkNet_" + args.dataset + "_final.pt")

            tqdm.write("epoch {:03d} ｜ acc:{:.2f} ｜ nmi:{:.2f} ｜ ari:{:.2f} ｜ f1:{:.2f}".format(epoch, acc, nmi, ari, f1))

            if args.wandb:
                wandb.log({"epoch": epoch, "loss": loss, "acc": acc, "nmi": nmi, "ari": ari, "f1": f1})
        else:

            if args.wandb:
                wandb.log({"epoch": epoch, "loss": loss})

    # testing
    model.load_state_dict(torch.load("./models/DinkNet_" + args.dataset + "_final.pt"))
    model.eval()

    y_hat = model.clustering(x, adj)

    acc, nmi, ari, f1 = evaluation(y, y_hat)
    tqdm.write("test      ｜ acc:{:.2f} ｜ nmi:{:.2f} ｜ ari:{:.2f} ｜ f1:{:.2f}".format(acc, nmi, ari, f1))


if __name__ == '__main__':

    # hyper-parameter settings
    parser = argparse.ArgumentParser("DinkNet")

    # data
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="training device")
    parser.add_argument("--dataset", type=str, default="citeseer", help="dataset name")

    # model
    parser.add_argument("--tradeoff", type=float, default=1e-10, help="tradeoff parameter")
    parser.add_argument("--activate", type=str, default="prelu", help="activation function")
    parser.add_argument("--hid_units", type=int, default=1536, help="number of hidden units")

    # training
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--wandb", type=bool, default=True, help="enable wandb")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--eval_inter", type=int, default=10, help="interval of evaluation")

    args = parser.parse_args()

    train(args=args)
