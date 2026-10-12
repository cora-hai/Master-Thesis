from model import MolNet, XtoCtoY, XtoC, FC, SideChannel, CtoY

import torch.nn as nn
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.nn import global_add_pool
import pandas as pd
import numpy as np
import io
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

from dig.xgraph.method import SubgraphX
from dig.xgraph.utils.init import fix_random_seed
fix_random_seed(27)

# Gemini
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


# full model with GNN encoder + CBM with only side channel logits as output
class GNN(nn.Module):
    def __init__(self, encoder: MolNet, cbm: XtoCtoY):
        super(GNN, self).__init__()
        self.encoder = encoder
        self.cbm = cbm

    def forward(self, x):
        embeddings = self.encoder(x)
        _, _, _, _, SC_probs = self.cbm(embeddings)
        return SC_probs


# Gemini
# wrapper model class to make our model compatible with SubgraphX explainer
class SubgraphXWrapper(nn.Module):

    def __init__(self, gnn_model):
        super(SubgraphXWrapper, self).__init__()
        self.gnn_model = gnn_model

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
        y_logits_sc = self.gnn_model.cbm.sc(embeddings)
        y_pred_sc = self.gnn_model.cbm.sigmoid(y_logits_sc)

        # 6. Format 2-class logits [log(1-p), log(p)] for DIG's internal softmax
        p_sc = torch.clamp(y_pred_sc.view(-1, 1), min=1e-7, max=1.0 - 1e-7)
        logits = torch.cat([torch.log(1.0 - p_sc), torch.log(p_sc)], dim=-1)

        return logits


# Gemini
def extract_node_importance(
    explanation_results, num_nodes, target_label_idx=0
):
    """Extracts per-node Shapley importance scores from SubgraphX results."""
    node_scores = np.zeros(num_nodes)

    # Access MCTS info list for target label
    mcts_info_list = explanation_results[target_label_idx]

    # Handle dictionary entries returned by DIG MCTS
    for info in mcts_info_list:
        if isinstance(info, dict):
            coalition = info.get("coalition", [])
            score = info.get("P", 0.0)
        else:
            coalition = getattr(info, "coalition", [])
            score = getattr(info, "P", 0.0)

        for node_idx in coalition:
            if node_idx < num_nodes:
                node_scores[node_idx] += score

    # Normalize scores to [0, 1] range for visualization
    score_range = node_scores.max() - node_scores.min()
    if score_range > 0:
        node_scores = (node_scores - node_scores.min()) / score_range

    return node_scores

# Gemini
def get_top_subgraph_smiles(
    smiles: str, explanation_results, top_k_atoms: int = 4
):
    """Extracts the top-K most important atoms as a SMILES fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_atoms = mol.GetNumAtoms()

    # Extract node scores using the fixed dictionary key access
    scores = extract_node_importance(explanation_results, num_nodes=num_atoms)

    # Get indices of top K atoms
    top_atom_indices = np.argsort(scores)[::-1][:top_k_atoms].tolist()

    # Extract matching sub-molecule bonds
    target_bonds = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in top_atom_indices
        and bond.GetEndAtomIdx() in top_atom_indices
    ]

    if target_bonds:
        sub_mol = Chem.PathToSubmol(mol, target_bonds)
        fragment_smiles = Chem.MolToSmiles(sub_mol)
    else:
        # Fallback if selected atoms don't share direct bonds
        fragment_smiles = f"Disconnected atom indices: {top_atom_indices}"

    # print(f"Top {top_k_atoms} Atom Indices: {top_atom_indices}")
    # print(f"Sub-SMILES Fragment: {fragment_smiles}")
    #return fragment_smiles, top_atom_indices
    return fragment_smiles


# Gemini
def visualize_subgraphx_explanation(
    smiles: str,
    explanation_results,
    target_label_idx: int = 0,
    save_path: str = None,
    image_size: tuple = (600, 400),
):
    """Maps SubgraphX importance scores onto RDKit 2D molecule layout

    using modern rdMolDraw2D atom highlighting.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_atoms = mol.GetNumAtoms()

    # 1. Extract per-node importance scores using dictionary key access
    atom_weights = extract_node_importance(
        explanation_results,
        num_nodes=num_atoms,
        target_label_idx=target_label_idx,
    )

    # 2. Print Atom-to-Score Mapping
    print("Atom Importance Mapping:")
    print("-" * 45)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        print(
            f"Atom Index {idx:2d} | Element: {symbol:2s} | Importance Score: {atom_weights[idx]:.4f}"
        )

    # 3. Compute continuous atom colors and radii based on Shapley importance
    # Use 'jet' or 'plasma' colormap
    colormap = cm.get_cmap("jet")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    atom_colors = {}
    atom_radii = {}

    for idx, weight in enumerate(atom_weights):
        # Assign color from map (RGBA tuple)
        rgba = colormap(norm(weight))
        atom_colors[idx] = (rgba[0], rgba[1], rgba[2])
        # Scale highlighting circle radius with importance
        atom_radii[idx] = 0.2 + 0.3 * float(weight)

    # 4. Generate highlighted 2D drawing via rdMolDraw2D
    drawer = rdMolDraw2D.MolDraw2DCairo(image_size[0], image_size[1])
    opts = drawer.drawOptions()
    opts.continuousHighlight = True
    opts.fillHighlights = True

    # Prepare molecule 2D coordinates
    mol_to_draw = Chem.Mol(mol)
    rdMolDraw2D.PrepareMolForDrawing(mol_to_draw)

    # Draw with atom highlights
    drawer.DrawMolecule(
        mol_to_draw,
        highlightAtoms=list(range(num_atoms)),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=[],  # Optional: add bond indices if desired
    )
    drawer.FinishDrawing()

    # 5. Convert drawing binary output to Image / Matplotlib figure
    png_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png_data))

    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title("SubgraphX Explanation (Side Channel Output)", fontsize=12)

    if save_path:
        img.save(save_path)
        print(f"\nExplanation visualization saved to: {save_path}")

    plt.show()
    return atom_weights


def main(args):

    # load model
    print("loading model...")
    encoder_trained = torch.load(args.encoder_file, weights_only = False, map_location = torch.device("cpu"))
    XtoCtoY_trained = torch.load(args.cbm_file, weights_only = False, map_location = torch.device("cpu"))
    gnn_model = GNN(encoder_trained, XtoCtoY_trained)
    explainer_model = SubgraphXWrapper(gnn_model)

    # define SubgraphX explainer
    subgraphx_explainer = SubgraphX(explainer_model,
                                    num_classes = 2,    # for binary classification
                                    device = "cpu",
                                    explain_graph = True,   # for graph classification (not node classification)
                                    reward_method = "mc_l_shapley",     # for graph classification with Shapley
                                    )

    # load data into dataset
    print("loading data...")
    df = pd.read_csv(args.data_file, header = 0)
    dataset = dataframe_to_pyg_dataset(df, smiles_col = "Drug", label_col = "Y")

    # get top substructure for each molecule in the dataset
    print("getting explanations...")
    top_substructures = []
    for i in range(len(dataset)):
        sample = dataset[i]
        #print(f"{sample = }")
        _, explanation_results, _ = subgraphx_explainer(x = sample.x, 
                                                        edge_index = sample.edge_index, 
                                                        edge_attr = sample.edge_attr, 
                                                        max_nodes = args.max_nodes)

        smiles_str = df["Drug"][i]
        #print(f"{smiles_str = }")
        fragment_smiles = get_top_subgraph_smiles(smiles = smiles_str, 
                                                  explanation_results = explanation_results, 
                                                  top_k_atoms = args.max_nodes)

        #print(f"{fragment_smiles}")
        top_substructures.append(fragment_smiles)

    df["Top Substructure"] = top_substructures
    print(df["Top Substructure"].value_counts())

    # write dataframe with most important substructure per molecule to csv file
    identifier = os.path.basename(args.data_file)
    df.to_csv(f"{args.outfolder}/{identifier.replace('.csv', '_feat-attr.csv')}", index = False, header = True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-file", type = str, help = "path to trained GNN encoder")
    ap.add_argument("--cbm-file", type = str, help = "path to trained CBM with side channel")
    ap.add_argument("--data-file", type = str, help = "path to csv file with data to explain")
    ap.add_argument("--outfolder", type = str, help = "path to directory to write output to")
    ap.add_argument("--max-nodes", type = int, nargs = "?", default = 5, help = "maximal number of nodes in output subgraph")
    args = ap.parse_args()

    main(args)