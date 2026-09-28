# models + training
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils.smiles import from_smiles
from sklearn.metrics import roc_auc_score, accuracy_score

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

    def forward(self, x):
        return self.linear(x)

# fully connected module for prediction of one concept
class FC(nn.Module):
    def __init__(self, input_dim) -> None:
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def binarise(self, x):
        classes = []
        for elem in x:
            if elem >= 0.5:
                classes.append(1)
            else:
                classes.append(0)
        return torch.tensor(classes)

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        x = self.binarise(x)
        return x

# X-C layer with 1 fully connected model per concept
class XtoC(nn.Module):
    def __init__(self, num_concepts, in_dims = 768):
        super(XtoC, self).__init__()
        self.all_fc = nn.ModuleList()
        for i in range(num_concepts):
            self.all_fc.append(FC(in_dims, 1))

    def forward(self, x):
        return [fc(x) for fc in self.all_fc]

# side channel
class SideChannel(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(SideChannel, self).__init__()
        ...

    def forward(self, x):
        ...

# full model
class XtoCtoY(nn.Module):
    def __init__(self, x2c, c2y, side_channel, num_concepts):
        super(XtoCtoY, self).__init__()
        self.x2c = x2c
        self.c2y = c2y
        self.sc = side_channel
        self.num_concepts = num_concepts

    def forward(self, x):
        # predict c
        c_preds_pre = self.x2c(x)
        c_preds = torch.transpose(torch.stack(c_preds_pre, dim = 0), 0, 1)
        c_preds = c_preds.reshape(-1, self.num_concepts)

        # predict y + append to concept predictions
        all_preds = [torch.stack([self.c2y(c_preds[i]) for i in range(c_preds.size(0))], dim = 0)]
        all_preds.append(c_preds_pre)

        return all_preds

# function for calling full model
def full_cbm(num_concepts, in_dims = 768) -> XtoCtoY:
    x2c_layer = XtoC(num_concepts = num_concepts, in_dims = in_dims)
    c2y_layer = CtoY(input_dim = num_concepts)
    return XtoCtoY(x2c = x2c_layer, c2y = c2y_layer, side_channel = None, num_concepts = num_concepts)

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

    ## load data ##
    DATA = {}
    DATA["train"] = pd.read_csv(f"{args.data_dir}/train.csv")
    DATA["val"] = pd.read_csv(f"{args.data_dir}/test.csv")
    DATA["test"] = pd.read_csv(f"{args.data_dir}/val.csv")


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


    ## initialise everything ##
    ModelXtoCtoY = full_cbm(num_concepts).to(DEVICE)
    encoder = MolNet(in_channels=DATA["train"]['Drug'][1].x.shape[1], hidden_channels=768).to(DEVICE)
    optimiser = torch.optim.Adam(list(encoder.parameters()) + list(ModelXtoCtoY.parameters()), lr = args.learning_rate)
    loss_c = torch.nn.BCELoss().to(DEVICE)
    loss_y = torch.nn.BCELoss().to(DEVICE)


    ## train & validation loop ##
    best_acc_score = -1
    for i in range(args.num_epochs):

        print(f"epoch {i+1}", flush = True)

        # training
        encoder.train()
        ModelXtoCtoY.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimiser.zero_grad()

            # run GNN encoding + CBM
            embeddings = encoder(batch)
            XtoC_output, XtoY_output = ModelXtoCtoY(embeddings)

            # loss calculation + backpropagation
            XtoC_output = torch.stack(XtoC_output, dim=1).squeeze()
            XtoC_loss = loss_c(torch.flatten(XtoC_output), batch.concepts.squeeze())
            XtoY_loss = loss_y(XtoY_output[0].squeeze(), batch.y.squeeze())
            joint_loss = XtoY_loss + XtoC_loss * args.loss_weight
            joint_loss.backward()
            optimiser.step()

            print(f"y loss = {XtoY_loss} | concept loss = {XtoC_loss}", flush = True)

        # validation
        encoder.eval()
        ModelXtoCtoY.eval()
        c_pred = np.array([])
        y_pred = np.array([])
        c_true = np.array([])
        y_true = np.array([])
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)

                # run GNN encoding + CBM
                embeddings = encoder(batch)
                XtoC_output, XtoY_output = ModelXtoCtoY(embeddings)

                # calculate concept + output accuracy
                y_true_batch = batch.y.cpu().numpy()
                y_true = np.append(y_true, y_true_batch)
                y_pred_batch = (XtoY_output[0].squeeze().cpu() > 0.5)
                y_pred = np.append(y_pred, y_pred_batch)
                y_acc = accuracy_score(y_true_batch, y_pred_batch)
                c_true_batch = batch.conceps.cpu().numpy()
                c_true = np.append(c_true, c_true_batch)
                c_pred_batch = XtoC_output
                c_pred = np.append(c_pred, c_pred_batch)
                c_acc = [accuracy_score(true, pred) for true, pred in zip(c_true_batch, c_pred_batch)]

                # record best validation accuracy & save best model config
                if y_acc > best_acc_score:
                    best_acc_score = y_acc
                    torch.save(encoder, f'{args.output_dir}/model_gnn_{args.data_type}_{args.selector}.pth')
                    torch.save(ModelXtoCtoY, f'{args.output_dir}/ModelXtoCtoY_layer_gnn_{args.data_type}_{args.selector}.pth')

                print(f"y accuracy = {y_acc} | concept accuracies = {c_acc}", flush = True)


    ## test loop ##
    encoder = torch.load(f'{args.output_dir}/model_gnn_{args.data_type}_{args.selector}.pth', weights_only = False)
    ModelXtoCtoY = torch.load(f'{args.output_dir}/ModelXtoCtoY_layer_gnn_{args.data_type}_{args.selector}.pth', weights_only = False)
    encoder.eval()
    ModelXtoCtoY.eval()
    c_pred = np.array([])
    y_pred = np.array([])
    c_true = np.array([])
    y_true = np.array([])
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)

            # run GNN encoding + CBM
            embeddings = encoder(batch)
            XtoC_output, XtoY_output = ModelXtoCtoY(embeddings)

            # get true and predicted concepts and outputs
            y_true = np.append(y_true, batch.y.cpu().numpy())
            y_pred = np.append(y_pred, (XtoY_output[0].squeeze().cpu() > 0.5))
            c_true = np.append(c_true, batch.conceps.cpu().numpy())
            c_pred = np.append(c_pred, XtoC_output)
        
    test_c_accs = [accuracy_score(true, pred) for true, pred in zip(c_true, c_pred)]
    test_y_acc = accuracy_score(y_true, y_pred)
    test_y_auroc = roc_auc_score(y_true, y_pred)
    print(f"test y accuracy = {test_y_acc}", flush = True)
    print(f"test y auroc = {test_y_auroc}", flush = True)
    print(f"test concept accuracies = {test_c_accs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type = str, help = "path to input data directory")
    ap.add_argument("--data-type", type = str, help = "name of dataset")
    ap.add_argument("--output-dir", type = str, help = "path to directory where outputs and logs will be saved")
    ap.add_argument("--selector", type = str, nargs = "?", default = "no", help = "concept selection method")
    ap.add_argument("--loss-weight", type = float, nargs = "?", default = 1.0, help = "weight for joint loss function")
    ap.add_argument("--learning-rate", type = float, nargs = "?", default = 2e-4, help = "learning rate for model optimisation")
    ap.add_argument("--num-epochs", type = int, nargs = "?", default = 200, help = "number of epochs to train for")
    args = ap.parse_args()

    train_and_evaluate(args)