from model import MolNet, XtoCtoY, XtoC, FC, SideChannel, CtoY, SideChannelXtoC

import torch
from torch import nn
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.nn import global_add_pool
#from torch_geometric.utils import subgraph
from rdkit import Chem

import pandas as pd
import numpy as np
import argparse
import os


def smiles_to_graph_data(smiles, y):
    """Converts a single SMILES string and label into atom and bond features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Skip invalid SMILES

    # 1. Node features (x): Atomic Numbers [num_atoms, 1]
    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # 2. Edge index (edge_index): Undirected graph connectivity [2, num_edges]
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if edge_indices:
        edge_index = (
            torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 3. Target label (y)
    y_tensor = torch.tensor([y], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y_tensor)


class SimplePyGDataset(InMemoryDataset):
    """Custom PyG InMemoryDataset wrapper."""

    def __init__(self, data, slices):
        super().__init__()
        self._data, self.slices = data, slices


def dataframe_to_pyg_dataset(df, smiles_col="smiles", label_col="Y"):
    """Transforms a DataFrame of SMILES into a PyG InMemoryDataset (unsplit)."""
    data_list = []

    for _, row in df.iterrows():
        data = smiles_to_graph_data(row[smiles_col], row[label_col])
        if data is not None:
            data_list.append(data)

    # Collate all graphs into a single PyG batch structure and slice index
    data, slices = InMemoryDataset.collate(data_list)

    # Instantiate and return the PyG InMemoryDataset object directly
    dataset = SimplePyGDataset(data, slices)
    return dataset


class GNN(nn.Module):
    def __init__(self, encoder: MolNet, cbm: XtoCtoY):
        super(GNN, self).__init__()
        self.encoder = encoder
        self.cbm = cbm

    def forward(self, x):
        embeddings = self.encoder(x)
        _, _, _, _, SC_probs = self.cbm(embeddings)
        return SC_probs

class Wrapper(nn.Module):

    def __init__(self, gnn_model, num_concepts, sc_out_dim):
        super(Wrapper, self).__init__()
        self.gnn_model = gnn_model
        self.sigmoid = nn.Sigmoid()
        self.num_concepts = num_concepts
        self.sc_out_dim = sc_out_dim

    def forward(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        batch=None,
        data=None,
        **kwargs,
    ):
        # 1. Unpack PyG Data/Batch object if DIG calls model(data=batch) or passes a single positional Data object
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            edge_attr = getattr(data, "edge_attr", None)
            batch = getattr(data, "batch", None)
        elif isinstance(x, (Data, Batch)):
            data_obj = x
            x = data_obj.x
            edge_index = data_obj.edge_index
            edge_attr = getattr(data_obj, "edge_attr", None)
            batch = getattr(data_obj, "batch", None)

        num_nodes = x.size(0)

        # 2. Assign default batch indices if missing (single graph)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        # 3. Handle missing edge_attr during MCTS pruning/subgraph evaluation
        num_edges = edge_index.size(1)
        if edge_attr is None or edge_attr.size(0) != num_edges:
            edge_attr = torch.zeros(
                (num_edges, 3), dtype=torch.float32, device=x.device
            )

        # 4. Forward pass through MolNet Encoder
        x_enc = self.gnn_model.encoder.conv1(
            x.to(torch.float32), edge_index, edge_attr.to(torch.float32)
        ).relu()
        x_enc = self.gnn_model.encoder.conv2(
            x_enc.to(torch.float32), edge_index, edge_attr.to(torch.float32)
        ).relu()

        # Global pooling -> graph embedding [batch_size, hidden_dim]
        embeddings = global_add_pool(x_enc, batch)

        # 5. Direct Side Channel execution (bypasses FC concept layers)
        c_logits_sc_pre = self.gnn_model.cbm.sc(embeddings)
        c_logits_sc = torch.transpose(torch.stack(c_logits_sc_pre, dim = 0), 0, 1)
        c_logits_sc = c_logits_sc.reshape(-1, self.sc_out_dim)
        c_preds_sc = self.sigmoid(c_logits_sc)
        padding_tensor = torch.zeros(c_preds_sc.size(0), self.num_concepts)
        y_logits_sc = self.gnn_model.cbm.c2y(torch.cat([padding_tensor, c_preds_sc], dim = -1))
        y_pred_sc = self.sigmoid(y_logits_sc)

        # # 6. Format 2-class logits [log(1-p), log(p)] for DIG's internal softmax
        # p_sc = torch.clamp(y_pred_sc.view(-1, 1), min=1e-7, max=1.0 - 1e-7)
        # logits = torch.cat([torch.log(1.0 - p_sc), torch.log(p_sc)], dim=-1)

        # return logits
        return y_pred_sc

def get_top_subgraph_smiles(smiles: str, explanation, top_k_atoms: int = 6) -> str:

    # convert molecule from smiles to molecular graph
    mol = Chem.MolFromSmiles(smiles)

    # extract top k nodes from explanation results
    scores = explanation.node_mask.view(-1)
    top_atom_indices = np.argsort(scores.numpy())[::-1][:top_k_atoms].tolist()
    
    #print(f"{top_atom_indices = }")

    # extract matching bonds
    target_bonds = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in top_atom_indices
        and bond.GetEndAtomIdx() in top_atom_indices
    ]

    #print(f"{target_bonds = }")

    # build subgraph from nodes and bonds
    if target_bonds:
        sub_mol = Chem.PathToSubmol(mol, target_bonds)
        fragment_smiles = Chem.MolToSmiles(sub_mol)
        if sub_mol.GetNumAtoms() != len(top_atom_indices):
            #print("discarding disconnected atoms...")
            pass
    elif top_k_atoms == 1:
        sub_mol = mol.GetAtomWithIdx(top_atom_indices[0])
        fragment_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse = top_atom_indices, bondsToUse = [], canonical = True)
    else:
        fragment_smiles = "_"

    return fragment_smiles

def main(args) -> None:

    # load CBM+SC model
    print("loading model...")
    encoder_trained = torch.load(args.encoder_file, weights_only = False, map_location = torch.device("cpu"))
    XtoCtoY_trained = torch.load(args.cbm_file, weights_only = False, map_location = torch.device("cpu"))
    gnn_model = GNN(encoder_trained, XtoCtoY_trained)
    explainer_model = Wrapper(gnn_model, num_concepts = args.num_concepts, sc_out_dim = args.sc_out_dim)

    # load data into dataset
    print("loading data...")
    df = pd.read_csv(args.data_file, header = 0)
    dataset = dataframe_to_pyg_dataset(df, smiles_col = "Drug", label_col = "Y")

    # define GNNExplainer model
    print("getting explanations...")
    explainer = Explainer(
        model = explainer_model,
        algorithm = GNNExplainer(epochs=args.explainer_epochs),
        explanation_type = "model",
        node_mask_type = "attributes",
        edge_mask_type = "object",
        model_config = dict(
            mode = "binary_classification",
            task_level = "graph",
            return_type = "probs"
        ), 
    )

    if args.max_nodes_range != -1:
        for m in range(1, args.max_nodes_range + 1):
            top_substructures = []
            for i in range(len(dataset)):
                sample = dataset[i]
                explanation = explainer(sample.x, sample.edge_index)
    
                smiles = df["Drug"][i]
                top_substructure = get_top_subgraph_smiles(smiles, explanation, top_k_atoms = m)
                top_substructures.append(top_substructure)
                print(f"{smiles = }")
                print(f"{top_substructure = }")
    
            df[f"Top Substructure {(m)}"] = top_substructures

    else:

        top_substructures = []
        for i in range(len(dataset)):
            sample = dataset[i]
            explanation = explainer(sample.x, sample.edge_index)

            smiles = df["Drug"][i]
            top_substructure = get_top_subgraph_smiles(smiles, explanation, top_k_atoms = args.max_nodes)
            top_substructures.append(top_substructure)
            print(f"{smiles = }")
            print(f"{top_substructure = }")

        df["Top Substructure"] = top_substructures


    outpath = os.path.join(args.outfolder, args.outfile)
    df.to_csv(outpath, index = False, header = True)


if __name__ == "__main__":

    encoder_file_default = "models/encoder_nitro_SC_27.pth"
    cbm_file_default = "models/ModelXtoCtoY_nitro_SC_27.pth"
    data_file_default = "data/mutag_dataset_remove_nitro/test_mutag_subgraphx.csv"
    outfile_default = "feature_attribution_out_gnnexplainer"

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-file", type = str, nargs = "?", default = encoder_file_default, help = "path to trained GNN encoder")
    ap.add_argument("--cbm-file", type = str, nargs = "?", default = cbm_file_default, help = "path to trained CBM with side channel")
    ap.add_argument("--data-file", type = str, nargs = "?", default = data_file_default, help = "path to csv file with data to explain")
    ap.add_argument("--num-concepts", type = int, nargs = "?", default = 15, help = "number of concepts in bottleneck")
    ap.add_argument("--sc-out-dim", type = int, nargs = "?", default = 6, help = "number of output nodes in side channel")
    ap.add_argument("--outfolder", type = str, nargs = "?", default = outfile_default, help = "path to directory to write output to")
    ap.add_argument("--outfile", type = str, help = "csv file to write results to")
    ap.add_argument("--explainer-epochs", type = int, nargs = "?", default = 200, help = "train explainer model for this many epochs")
    ap.add_argument("--max-nodes", type = int, nargs = "?", default = 6, help = "maximal number of nodes in extracted subgraph")
    ap.add_argument("--max-nodes-range", type = int, nargs = "?", default = -1, help = "switch on range of max nodes for experiments")
    args = ap.parse_args()

    main(args)