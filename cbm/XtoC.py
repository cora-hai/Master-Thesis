# models + training
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils.smiles import from_smiles
from sklearn.metrics import roc_auc_score, accuracy_score, jaccard_score, f1_score
from torchvision.ops import StochasticDepth # for side channel dropout

# utils
import argparse
import pandas as pd 
import numpy as np 
from utils import set_seed
#import wandb


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset for SMILES input
class CBMDataset(Dataset):
    def __init__(self, split, features, tokenizer, DATA):
        self.data = DATA[split]
        self.features = features
        self.tokenizer = tokenizer
        self.labels = self.data['Y']
        self.text = self.data["Drug"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        tokenized_text = self.tokenizer(
            self.text[index], 
            max_length=64,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'input_ids': tokenized_text['input_ids'],
            'attention_mask': tokenized_text['attention_mask'],
            'label': torch.tensor(self.labels[index], dtype=torch.int),
            'concept_labels': self.data.iloc[index][self.features].values.astype(int),
            'features': self.features
        }

# GNN embedding model
class MolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MolNet, self).__init__()

        # 3 is the number of edge features from the 'from_smiles' function
        nn1 = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn1, edge_dim=3)
        self.conv2 = GINEConv(nn2, edge_dim=3)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x.to(torch.float32), edge_index, edge_attr.to(torch.float32)).relu()
        x = self.conv2(x.to(torch.float32), edge_index, edge_attr.to(torch.float32)).relu()
        x = global_add_pool(x, batch)
        return x

# fully connected module for prediction of one concept
class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim = 768*2) -> None:
        super(FC, self).__init__()
        self.stack = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return self.stack(x)

# X-C layer with 1 fully connected model per concept
class XtoC(nn.Module):
    def __init__(self, num_concepts, in_dims = 768):
        super(XtoC, self).__init__()
        self.all_fc = nn.ModuleList()
        for _ in range(num_concepts):
            self.all_fc.append(FC(in_dims))

    def forward(self, x):
        return [fc(x) for fc in self.all_fc]

def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    # gradient approximation for binary concept neurons, based on
    # 1. https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/functional.py#L2219
    # 2. https://github.com/AngelosNal/PyTorch-Gumbel-Sigmoid/tree/main
    
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
        
# full model
class XtoCtoY(nn.Module):
    def __init__(self, x2c, num_concepts):
        super(XtoCtoY, self).__init__()
        self.x2c = x2c
        self.num_concepts = num_concepts
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # predict c with Gumbel Sigmoid for handling gradient approximation
        c_logits_pre = self.x2c(x)
        c_logits = torch.transpose(torch.stack(c_logits_pre, dim = 0), 0, 1)
        c_logits = c_logits.reshape(-1, self.num_concepts)
        #c_preds = self.sigmoid(c_logits)
        c_preds = gumbel_sigmoid(c_logits, hard = True)

        return c_logits_pre, c_preds

# function for calling full model
def full_cbm(num_concepts, in_dims = 768) -> XtoCtoY:
    x2c_layer = XtoC(num_concepts = num_concepts, in_dims = in_dims)
    return XtoCtoY(x2c = x2c_layer, num_concepts = num_concepts)

# molecules are PyG objects, so we need to attach the y and concepts to the object
def attach_y_and_concepts(row, features):
    row['Drug'].y = torch.tensor(row['Y'], dtype=torch.float32)
    row['Drug'].target_names = row['Drug_ID']
    try:
        row['Drug'].concepts = torch.tensor(np.asarray(row[features].values, dtype=np.float32))
    except:
        pass
    return row['Drug']


##########################
## train & evaluate CBM 
##########################

def train_and_evaluate(args):

    #print(args.loss_weight)

    ## load data ##
    DATA = {}
    DATA["train"] = pd.read_csv(f"{args.data_dir}/train_{args.data_type}.csv")
    DATA["val"] = pd.read_csv(f"{args.data_dir}/val_{args.data_type}.csv")
    DATA["test"] = pd.read_csv(f"{args.data_dir}/test_{args.data_type}.csv")

    # get name and number of concepts
    features = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']).columns.to_list()
    print(features)
    num_concepts = len(features)

    ## prepare data ##

    # turn the SMILES strings into PyG objects
    DATA["train"]['Drug'] = DATA["train"]['Drug'].apply(from_smiles)
    DATA["val"]['Drug'] = DATA["val"]['Drug'].apply(from_smiles)
    DATA["test"]['Drug'] = DATA["test"]['Drug'].apply(from_smiles)  

    # attach the y and concepts to the PyG objects
    DATA["train"]['Drug'] = DATA["train"].apply(attach_y_and_concepts, args = (features,), axis=1)
    DATA["val"]['Drug'] = DATA["val"].apply(attach_y_and_concepts, args = (features,), axis=1)
    DATA["test"]['Drug'] = DATA["test"].apply(attach_y_and_concepts, args = (features,), axis=1)

    # create the data loaders
    train_loader = DataLoader(DATA["train"]['Drug'], batch_size=32, shuffle=True)
    val_loader = DataLoader(DATA["val"]['Drug'], batch_size=32, shuffle=False)
    test_loader = DataLoader(DATA["test"]['Drug'], batch_size=32, shuffle=False)


    # alternative to different weights per class:
    pos_weight_tensor = torch.tensor([5.0]).to(DEVICE)

    ## initialise everything ##
    ModelXtoCtoY = full_cbm(args.num_concepts).to(DEVICE)
    encoder = MolNet(in_channels=DATA["train"]['Drug'][1].x.shape[1], hidden_channels=768).to(DEVICE)
    optimiser = torch.optim.Adam(list(encoder.parameters()) + list(ModelXtoCtoY.parameters()), lr = args.learning_rate)
    loss_c = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight_tensor).to(DEVICE)  # weighted concept loss

    ## configure W&B logging
    # config = args.__dict__
    # run = wandb.init(project = "CBM", config = config)
    # run.watch(ModelXtoCtoY, loss_y, log = "all", log_freq = 10)

    ## train & validation loop ##
    best_acc_score = -1
    for i in range(args.num_epochs):

        # training
        encoder.train()
        ModelXtoCtoY.train()
        c_loss_list = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimiser.zero_grad()

            # run GNN encoding + CBM
            embeddings = encoder(batch)
            XtoC_logits, _ = ModelXtoCtoY(embeddings)

            # loss calculation + backpropagation
            XtoC_logits = torch.stack(XtoC_logits, dim=1).squeeze()
            XtoC_loss = loss_c(XtoC_logits, batch.concepts.float().view(-1, num_concepts))
            XtoC_loss.backward()
            optimiser.step()

            c_loss_list.append(XtoC_loss.item())
      
        c_loss_mean = np.mean(c_loss_list)
        
        #run.log({"epoch": i+1, "Y loss": y_loss_mean, "C loss": c_loss_mean})

        # if (i+1) % 5 == 0:
        #     print(f"epoch {i + 1} | Y loss = {y_loss_mean} | C loss = {c_loss_mean} | joint loss = {joint_loss_mean}", flush = True)

        # validation
        encoder.eval()
        ModelXtoCtoY.eval()
        c_pred = []
        c_true = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)

                # run GNN encoding + CB
                embeddings = encoder(batch)
                XtoC_logits, XtoC_preds = ModelXtoCtoY(embeddings)

                # get concept predictions per batch
                c_true.append(batch.concepts.cpu().numpy().reshape(-1, num_concepts))
                c_pred_binary = (XtoC_preds.cpu() > 0.5).float()
                c_pred.append(c_pred_binary)

        c_true_all = np.vstack(c_true)
        c_pred_all = np.vstack(c_pred)
        c_acc = (c_pred_all == c_true_all).mean(axis = 0).tolist()
        c_acc_mean = np.mean(c_acc)

        if (i+1) % 10 == 0:
            c_jaccard = jaccard_score(c_true_all, c_pred_all, average = None, zero_division = 0.0)
            c_jaccard_mean = np.mean(c_jaccard)
            c_f1 = f1_score(c_true_all, c_pred_all, average = None, zero_division = 0)
            c_f1_mean = np.mean(c_f1)

        # record best validation accuracy & save best model config
        if c_acc_mean > best_acc_score:
            best_acc_score = c_acc_mean
            #wandb.unwatch()
            torch.save(encoder, f'{args.output_dir}/encoder_gnn_{args.data_type}.pth')
            torch.save(ModelXtoCtoY, f'{args.output_dir}/ModelXtoC_gnn_{args.data_type}.pth')


        if (i+1) % 10 == 0:
            print(f"epoch {i+1} | mean concept acc = {c_acc_mean} | mean concept jaccard = {c_jaccard_mean} | mean concept f1 = {c_f1_mean}", flush = True)
        # print(f"concept accuracies = {c_acc}", flush = True)
        # print(f"concept jaccards = {c_jaccard}", flush = True)


    ## test loop ##
    encoder = torch.load(f'{args.output_dir}/encoder_gnn_{args.data_type}.pth', weights_only = False)
    ModelXtoCtoY = torch.load(f'{args.output_dir}/ModelXtoC_gnn_{args.data_type}.pth', weights_only = False)
    # encoder = best_encoder
    # ModelXtoCtoY = best_ModelXtoCtoY
    encoder.eval()
    ModelXtoCtoY.eval()
    c_pred = []
    c_true = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)

            # run GNN encoding + CBM
            embeddings = encoder(batch)
            XtoC_logits, XtoC_preds = ModelXtoCtoY(embeddings)

            # get concept and output predictions per batch
            c_true.append(batch.concepts.cpu().numpy().reshape(-1, num_concepts))
            c_pred_binary = (XtoC_preds.cpu() > 0.5).float()
            c_pred.append(c_pred_binary)

    # calculate concept and output accuracies
    c_true_all = np.vstack(c_true)
    c_pred_all = np.vstack(c_pred)
    test_c_acc = (c_pred_all == c_true_all).mean(axis = 0).tolist()
    test_c_acc_mean = np.mean(test_c_acc)
    test_c_jaccard = jaccard_score(c_true_all, c_pred_all, average = None, zero_division = 0.0)
    test_c_jaccard_mean = np.mean(test_c_jaccard)
    test_c_f1 = f1_score(c_true_all, c_pred_all, average = None, zero_division = 0)
    test_c_f1_mean = np.mean(test_c_f1)
      
    print(f"test mean concept acc = {test_c_acc_mean}", flush = True)
    print(f"test mean concept jaccard = {test_c_jaccard_mean}", flush = True)
    print(f"test mean concept F1 = {test_c_f1_mean}", flush = True)
    print(f"test concept accuracies = {test_c_acc}", flush = True)
    print(f"test concept jaccards = {list(test_c_jaccard)}", flush = True)
    print(f"test concept F1 = {list(test_c_f1)}", flush = True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type = str, help = "path to input data directory")
    ap.add_argument("--data-type", type = str, help = "name of dataset")
    ap.add_argument("--output-dir", type = str, help = "path to directory where outputs and logs will be saved")
    ap.add_argument("--learning-rate", type = float, nargs = "?", default = 2e-4, help = "learning rate for model optimisation")
    ap.add_argument("--num-concepts", type = int, nargs = "?", default = 21, help = "number of concepts in the concept bottleneck")
    ap.add_argument("--num-epochs", type = int, nargs = "?", default = 200, help = "number of epochs to train for")
    ap.add_argument("--seed", type = int, nargs = "?", default = 27, help = "random seed for reproducibility")
    args = ap.parse_args()

    #wandb.login()

    set_seed(args.seed)

    train_and_evaluate(args)