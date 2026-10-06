# models + training
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils.smiles import from_smiles
from sklearn.metrics import roc_auc_score, accuracy_score, jaccard_score, f1_score
from torchvision.ops import StochasticDepth # for side channel dropout

# concept selection
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from greedy_set_cover import get_cover
from fpmax import run_fpmax

# utils
import argparse
import pandas as pd 
import numpy as np 
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

# linear C-Y layer
class CtoY(nn.Module):
    def __init__(self, input_dim) -> None:
        super(CtoY, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        #x = self.sigmoid(x)
        return x

# fully connected module for prediction of one concept
class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim = 768*2) -> None:
        super(FC, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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

# side channel with one hidden layer + ReLU activation
class SideChannel(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 768, out_dim = 1, p_drop = 1.0) -> None:
        super(SideChannel, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            StochasticDepth(p = p_drop, mode = "batch")
        )

    def forward(self, x):
        return self.stack(x)
        

# full model
class XtoCtoY(nn.Module):
    def __init__(self, x2c, c2y, side_channel, num_concepts):
        super(XtoCtoY, self).__init__()
        self.x2c = x2c
        self.c2y = c2y
        self.sc = side_channel
        self.num_concepts = num_concepts
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # predict c
        c_logits_pre = self.x2c(x)
        c_logits = torch.transpose(torch.stack(c_logits_pre, dim = 0), 0, 1)
        c_logits = c_logits.reshape(-1, self.num_concepts)
        c_probs = self.sigmoid(c_logits)

        # get side channel output
        y_logits_sc = self.sc(x)

        # get CBM output
        y_logits_cbm = torch.stack([self.c2y(c_probs[i]) for i in range(c_probs.size(0))], dim = 0)

        # get joint prediction
        y_pred = self.sigmoid(y_logits_cbm + y_logits_sc)
        y_pred_cbm = self.sigmoid(y_logits_cbm)
        y_pred_sc = self.sigmoid(y_logits_sc)

        return [y_pred, c_logits_pre, c_probs, y_pred_cbm, y_pred_sc]

# function for calling full model
def full_cbm(num_concepts, in_dims = 768) -> XtoCtoY:
    x2c_layer = XtoC(num_concepts = num_concepts, in_dims = in_dims)
    c2y_layer = CtoY(input_dim = num_concepts)
    side_channel = SideChannel(input_dim = in_dims, p_drop = args.dropout_p)
    return XtoCtoY(x2c = x2c_layer, c2y = c2y_layer, side_channel = side_channel, num_concepts = num_concepts)

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


    ## select concepts ##
    if args.selector == "l1":
        # according to https://scikit-learn.org/stable/modules/feature_selection.html#l1-based-feature-selection
        X, y = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']), DATA["train"]["Y"]

        # train linear support vector classifier with L1 penalty for "feature selection"
        lsvc = LinearSVC(C=0.01, penalty = "l1", dual = False).fit(X,y)     # C = regularisation parameter, strength inversely proportional to C
        selector = SelectFromModel(lsvc, prefit = True)
    
        # get selected features
        feature_mask = selector.get_support()
        features = X.columns[feature_mask].tolist()
        num_concepts = len(features)

        print(f"features = {features}")
        print(f"# chosen concepts = {num_concepts} out of {len(X.columns)}")

    elif args.selector == "tree":
        # according to https://scikit-learn.org/stable/modules/feature_selection.html#tree-based-feature-selection
        X, y = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']), DATA["train"]["Y"]
        print(f"X = {X.shape}")

        # maybe max_features nutzen für feste Anzahl an concepts?
        clf = ExtraTreesClassifier(n_estimators = 30, random_state = 42).fit(X,y)  # n_estimators = number of trees in the forest
        selector = SelectFromModel(clf, prefit = True)
        feature_mask = selector.get_support()
        features = X.columns[feature_mask].tolist()
        num_concepts = len(features)

        print(f"features = {features}")
        print(f"# chosen concepts = {num_concepts} out of {len(X.columns)}")

    elif args.selector == "gsc":
        X = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID'])
        features = get_cover(X)
        num_concepts = len(features)

        print(f"features = {features}")
        print(f"# chosen concepts = {num_concepts} out of {len(X.columns)}")

    elif args.selector == "fpmax":
        X = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID'])
        features = run_fpmax(X, min_support = 0.5)
        num_concepts = len(features)

        print(f"features = {features}")
        print(f"# chosen concepts = {num_concepts} out of {len(X.columns)}")

    elif args.selector == "no":
        X, y = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']), DATA["train"]["Y"]
        features = X.columns.to_list()
        num_concepts = len(features)

        print(f"# chosen concepts = {num_concepts} out of {len(X.columns)}")

    else:
        print(f"choose a valid concept selector method since {args.selector} is not")


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


    ## dynamic per-concept class imbalance weighting ##

    # # compute per-concept positive weights from training data
    # train_concepts = np.vstack([row[features].values.astype(float) for _, row in DATA["train"].iterrows()])
    # pos_counts = train_concepts.sum(axis=0)
    # neg_counts = len(train_concepts) - pos_counts

    # # calculate ratio (neg / pos) per concept with epsilon to prevent div-by-zero
    # pos_weight_values = np.where(pos_counts > 0, neg_counts / (pos_counts + 1e-5), 1.0)
    # pos_weight_tensor = torch.tensor(pos_weight_values, dtype=torch.float32).to(DEVICE)
    # print(f"{pos_weight_values = }")

    # alternative to different weights per class:
    pos_weight_tensor = torch.tensor([5.0]).to(DEVICE)


    ## initialise everything ##
    ModelXtoCtoY = full_cbm(num_concepts).to(DEVICE)
    encoder = MolNet(in_channels=DATA["train"]['Drug'][1].x.shape[1], hidden_channels=768).to(DEVICE)
    optimiser = torch.optim.Adam(list(encoder.parameters()) + list(ModelXtoCtoY.parameters()), lr = args.learning_rate)
    loss_c = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight_tensor).to(DEVICE)  # weighted concept loss
    loss_y = torch.nn.BCELoss().to(DEVICE)  # X-C-Y loss

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
        y_loss_list = []
        c_loss_list = []
        joint_loss_list = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimiser.zero_grad()

            # run GNN encoding + CBM
            embeddings = encoder(batch)
            XtoY_output, XtoC_logits, _, _, _ = ModelXtoCtoY(embeddings)

            # loss calculation + backpropagation
            XtoC_logits = torch.stack(XtoC_logits, dim=1).squeeze()
            XtoC_loss = loss_c(XtoC_logits, batch.concepts.float().view(-1, num_concepts))
            XtoY_loss = loss_y(XtoY_output.squeeze(), batch.y.squeeze())
            joint_loss = XtoY_loss + XtoC_loss * args.loss_weight
            joint_loss.backward()
            optimiser.step()

            y_loss_list.append(XtoY_loss.item())
            c_loss_list.append(XtoC_loss.item())
            joint_loss_list.append(joint_loss.item())

            #print(f"y loss = {XtoY_loss} | c loss = {XtoC_loss} | s loss = {SC_loss}", flush = True)
        
        y_loss_mean = np.mean(y_loss_list)
        c_loss_mean = np.mean(c_loss_list)
        joint_loss_mean = np.mean(joint_loss_list)

        #run.log({"epoch": i+1, "Y loss": y_loss_mean, "C loss": c_loss_mean})

        # if (i+1) % 5 == 0:
        #     print(f"epoch {i + 1} | Y loss = {y_loss_mean} | C loss = {c_loss_mean} | joint loss = {joint_loss_mean}", flush = True)

        # validation
        encoder.eval()
        ModelXtoCtoY.eval()
        c_pred = []
        y_pred = []
        c_true = []
        y_true = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)

                # run GNN encoding + CBM
                embeddings = encoder(batch)
                XtoY_output, _, XtoC_output, _, _ = ModelXtoCtoY(embeddings)

                # get concept and output predictions per batch
                y_true.extend(batch.y.cpu())
                y_pred.extend((XtoY_output.squeeze().cpu() > 0.5).float())
                c_true.append(batch.concepts.cpu().numpy().reshape(-1, num_concepts))
                c_pred_binary = (XtoC_output.cpu() > 0.5).float()
                c_pred.append(c_pred_binary)

        # calculate concept and output accuracies (most of the only every 10th epoch)
        y_acc = accuracy_score(y_true, y_pred)

        if (i+1) % 10 == 0:
            c_true_all = np.vstack(c_true)
            c_pred_all = np.vstack(c_pred)
            y_auroc = roc_auc_score(y_true, y_pred)
            y_jaccard = jaccard_score(y_true, y_pred, zero_division = 0.0)
            c_acc = (c_pred_all == c_true_all).mean(axis = 0).tolist()
            c_acc_mean = np.mean(c_acc)
            c_jaccard = jaccard_score(c_true_all, c_pred_all, average = None, zero_division = 0.0)
            c_jaccard_mean = np.mean(c_jaccard)
            c_f1 = f1_score(c_true_all, c_pred_all, average = None, zero_division = 0)
            c_f1_mean = np.mean(c_f1)

        # record best validation accuracy & save best model config
        if y_acc > best_acc_score:
            best_acc_score = y_acc
            #wandb.unwatch()
            torch.save(encoder, f'{args.output_dir}/encoder_gnn_{args.data_type}_{args.selector}_{args.loss_weight}.pth')
            torch.save(ModelXtoCtoY, f'{args.output_dir}/ModelXtoCtoY_gnn_{args.data_type}_{args.selector}_{args.loss_weight}.pth')
            # best_encoder = encoder
            # best_ModelXtoCtoY = ModelXtoCtoY

        #run.log({"epoch": i+1, "Y val acc": y_acc, "Y val auroc": y_auroc, "Y val jaccard": y_jaccard, "C val mean acc": c_acc_mean, "C val jaccard": c_jaccard_mean, "C val F1": c_f1_mean})

        if (i+1) % 10 == 0:
            print(f"epoch {i+1} | y accuracy = {y_acc} | y auroc = {y_auroc} | y jaccard = {y_jaccard}")
            print(f"epoch {i+1} | mean concept acc = {c_acc_mean} | mean concept jaccard = {c_jaccard_mean} | mean concept f1 = {c_f1_mean}", flush = True)
        # print(f"concept accuracies = {c_acc}", flush = True)
        # print(f"concept jaccards = {c_jaccard}", flush = True)

    # save best models
    # torch.save(best_encoder, f'{args.output_dir}/encoder_gnn_{args.data_type}_{args.selector}_{args.loss_weight}.pth')
    # torch.save(best_ModelXtoCtoY, f'{args.output_dir}/ModelXtoCtoY_gnn_{args.data_type}_{args.selector}_{args.loss_weight}.pth')
    # print(f"best model from epoch {best_epoch}")

    ## test loop ##
    encoder = torch.load(f'{args.output_dir}/encoder_gnn_{args.data_type}_{args.selector}_{args.loss_weight}.pth', weights_only = False)
    ModelXtoCtoY = torch.load(f'{args.output_dir}/ModelXtoCtoY_gnn_{args.data_type}_{args.selector}_{args.loss_weight}.pth', weights_only = False)
    # encoder = best_encoder
    # ModelXtoCtoY = best_ModelXtoCtoY
    encoder.eval()
    ModelXtoCtoY.eval()
    c_pred = []
    y_pred = []
    y_pred_cbm = []
    y_pred_sc = []
    c_true = []
    y_true = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)

            # run GNN encoding + CBM
            embeddings = encoder(batch)
            XtoY_output, _, XtoC_output, CtoY_probs, SC_probs = ModelXtoCtoY(embeddings)

            # get concept and output predictions per batch
            y_true.extend(batch.y.cpu())
            y_pred.extend((XtoY_output.squeeze().cpu() > 0.5).float())
            y_pred_cbm.extend((CtoY_probs.squeeze().cpu() > 0.5).float())
            y_pred_sc.extend((SC_probs.squeeze().cpu() > 0.5).float())
            c_true.append(batch.concepts.cpu().numpy().reshape(-1, num_concepts))
            c_pred_binary = (XtoC_output.cpu() > 0.5).float()
            c_pred.append(c_pred_binary)

    # calculate concept and output accuracies
    c_true_all = np.vstack(c_true)
    c_pred_all = np.vstack(c_pred)
    test_y_acc = accuracy_score(y_true, y_pred)
    test_y_acc_cbm = accuracy_score(y_true, y_pred_cbm)
    test_y_acc_sc = accuracy_score(y_true, y_pred_sc)
    test_y_auroc = roc_auc_score(y_true, y_pred)
    test_y_jaccard = jaccard_score(y_true, y_pred, zero_division = 0.0)
    test_c_acc = (c_pred_all == c_true_all).mean(axis = 0).tolist()
    test_c_acc_mean = np.mean(c_acc)
    test_c_jaccard = jaccard_score(c_true_all, c_pred_all, average = None, zero_division = 0.0)
    test_c_jaccard_mean = np.mean(test_c_jaccard)
    test_c_f1 = f1_score(c_true_all, c_pred_all, average = None, zero_division = 0)
    test_c_f1_mean = np.mean(c_f1)
      
    print(f"test y accuracy \t {test_y_acc}", flush = True)
    print(f"test y accuracy CBM \t {test_y_acc_cbm}", flush = True)
    print(f"test y accuracy SC \t {test_y_acc_sc}", flush = True)
    print(f"test y auroc \t {test_y_auroc}", flush = True)
    print(f"test y jaccard \t {test_y_jaccard}")
    print(f"test mean concept acc \t {test_c_acc_mean}", flush = True)
    print(f"test mean concept jaccard \t {test_c_jaccard_mean}", flush = True)
    print(f"test mean concept F1 \t {test_c_f1_mean}", flush = True)
    print(f"test concept accuracies = {test_c_acc}", flush = True)
    print(f"test concept jaccards: {list(test_c_jaccard)}", flush = True)
    print(f"test concept F1: {list(test_c_f1)}", flush = True)

    # get weights per concept
    concept_weights = ModelXtoCtoY.c2y.linear.weight.detach().cpu().numpy()
    concept_weights_df = pd.DataFrame(concept_weights, columns = features)
    weight_file = f"{args.output_dir}/ModelXtoCtoY_gnn_{args.data_type}_{args.selector}_concept_weights.csv"
    concept_weights_df.to_csv(weight_file, index = False)

    return {"Y test acc": test_y_acc, "Y test auroc": test_y_auroc, "C test acc": test_c_acc_mean}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type = str, help = "path to input data directory")
    ap.add_argument("--data-type", type = str, help = "name of dataset")
    ap.add_argument("--output-dir", type = str, help = "path to directory where outputs and logs will be saved")
    ap.add_argument("--selector", type = str, nargs = "?", default = "no", help = "concept selection method")
    ap.add_argument("--loss-weight", type = float, nargs = "?", default = 1.0, help = "weight for joint loss function")
    ap.add_argument("--learning-rate", type = float, nargs = "?", default = 2e-4, help = "learning rate for model optimisation")
    ap.add_argument("--dropout-p", type = float, nargs = "?", default = 1.0, help = "dropout probability for side channel regularisation")
    ap.add_argument("--num-epochs", type = int, nargs = "?", default = 200, help = "number of epochs to train for")
    args = ap.parse_args()

    #wandb.login()

    # logs = {}
    # for i in range(1, 21):
    #     lw = i/10
    #     args.loss_weight = lw
    #     logs[str(lw)] = train_and_evaluate(args)

    # with open("logs_out_no-sc.txt", "w", encoding = "utf-8") as f:
    #     f.write(str(logs))

    train_and_evaluate(args)