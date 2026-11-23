import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
import json
import argparse
import os

def load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding = "utf-8") as f:
        return json.load(f)

def get_counts(df: pd.DataFrame, i: int = -1) -> pd.Series:
    df_pos = df[df["Y"] == 1]
    if i == -1:
        return df_pos["Top Substructure"].value_counts()
    else:
        return df_pos[f"Top Substructure {i}"].value_counts()

def get_special(df: pd.DataFrame, i_max: int = -1) -> pd.DataFrame:
    df_pos = df[df["Y"] == 1]
    if i_max == -1:
        return df_pos[df_pos.drop(columns = ["Drug_ID", "Drug", "Y", "Top Substructure"]).eq(0).all(axis = 1)].loc[:,["Drug_ID", "Drug", "Y", "Top Substructure"]]
    else:
        cols = ["Drug_ID", "Drug", "Y"] + [f"Top Substructure {i}" for i in range(1, i_max + 1)]
        return df_pos[df_pos.drop(columns = cols).eq(0).all(axis = 1)].loc[:, cols]


def visualise_substructures(top_counts: pd.Series, outfolder: str, min_count: int = 1) -> None:

    idx = 0
    for smarts, count in top_counts.items():
        idx += 1

        if count < min_count:
            continue

        # get molecular graph of substructure
        sub = Chem.MolFromSmarts(smarts)
        if sub == None:
            continue

        # set atom label to "C" for all carbon atoms
        for atom in sub.GetAtoms():
            if atom.GetSymbol() == "C":
                atom.SetProp("atomLabel", "C")

        # configure drawer
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(200, 100)
        dopts = drawer.drawOptions()
        dopts.bondLineWidth = 2.0
        dopts.addAtomIndices = False
        dopts.clearBackground = False

        # draw substructure
        drawer.DrawMolecule(sub, legend = f"{smarts} (# = {count})")
        drawer.FinishDrawing()

        # save drawing to svg file
        svg_text = drawer.GetDrawingText()
        filename = os.path.join(outfolder, f"{idx}_{smarts}.svg")
        with open(filename, "w") as f:
            f.write(svg_text)


def draw_highlighted_molecules(df: pd.DataFrame, outfolder: str, i: int = -1) -> None:

    for _, row in df.iterrows():
        molecule = row["Drug"]
        if i == -1:
            substructure = row["Top Substructure"]
        else:
            substructure = row[f"Top Substructure {i}"]
        drug_id = row["Drug_ID"]

        # get molecule graphs for molecule and substructure
        mol = Chem.MolFromSmiles(molecule)
        sub = Chem.MolFromSmarts(substructure)

        # set atom label to "C" for all carbon atoms
        for atom in sub.GetAtoms():
            if atom.GetSymbol() == "C":
                atom.SetProp("atomLabel", "C")

        # set atom label to "C" for all carbon atoms
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C":
                atom.SetProp("atomLabel", "C")

        # generate 2D coordinates if not present
        if not mol.GetNumConformers():
            rdMolDraw2D.PrepareMolForDrawing(mol)

        # find matching atoms and bonds
        matches = mol.GetSubstructMatch(sub)

        # initialise drawer
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 400)
        dopts = drawer.drawOptions()
        dopts.bondLineWidth = 2.0
        dopts.addAtomIndices = False
        dopts.clearBackground = False

        # draw molecule with highlighted substructure
        drawer.DrawMolecule(mol, highlightAtoms = matches)
        drawer.FinishDrawing()

        # save drawing to svg file
        svg_text = drawer.GetDrawingText()
        filename = os.path.join(outfolder, f"{drug_id}.svg")
        with open(filename, "w") as f:
            f.write(svg_text)

def check(substructure: str, concepts: dict, use_qq = True) -> list:

    concept_mols = {key : Chem.MolFromSmarts(val) for key, val in concepts.items()}
    sub = Chem.MolFromSmarts(substructure)

    matches = []
    for key, val in concept_mols.items():
        if sub is not None:
            if sub.HasSubstructMatch(val, useQueryQueryMatches = use_qq):
                matches.append(key)

    return matches

def check_all_substructures(substructures: list, concepts: dict, use_qq = True):
    matches_per_substruct = {}
    match_count = 0
    for s in substructures:
        if "[" in s and "," in s:
            continue
        matches_per_substruct[s] = check(s, concepts, use_qq)
        if matches_per_substruct[s] != []:
            match_count += 1

    return match_count, matches_per_substruct

def check_deleted_against_identified(deleted: dict, deleted_matches: dict):

    deleted_matches_list = []
    for m in deleted_matches:
        deleted_matches_list.extend(deleted_matches[m])
    total = len(deleted)
    found_count = 0
    found = {}
    for concept in deleted.keys():
        if concept in set(deleted_matches_list):
            found_count += 1
            found[concept] = 0

    for match in deleted_matches_list:
        found[match] += 1

    return f"{found_count}/{total}", found


def main(args):

    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)

    # read SubgraphX results
    df = pd.read_csv(args.subgraphx_file, header = 0)
    print(f"Eval for {args.subgraphx_file} \n")

    # get cooccurrence counts with positive class for every substructure
    cooc_counts = get_counts(df)
    top_substructures = [idx for idx in cooc_counts.index if "," not in idx]
    print(f"Substructures by cooccurrence count:\n{cooc_counts}\n")

    # get drawings of identified substructures
    outfolder_drawings = os.path.join(args.outfolder, "substructure_drawings")
    if not os.path.exists(outfolder_drawings):
        os.makedirs(outfolder_drawings)
    visualise_substructures(cooc_counts, outfolder = outfolder_drawings, min_count = 2)

    # get positive molecules with all positive concepts deleted
    special_molecules = get_special(df)
    print(f"Special molecules:\n{special_molecules}\n")

    # for special molecules, draw molecule with highlighted substructure
    outfolder_special = os.path.join(args.outfolder, "special_highlights")
    if not os.path.exists(outfolder_special):
        os.makedirs(outfolder_special)
    draw_highlighted_molecules(special_molecules, outfolder = outfolder_special)

    # check identified substructures against deleted concepts
    deleted = load_json(args.deleted_concepts)
    deleted_matches = check_all_substructures(top_substructures, deleted, use_qq = False)
    print(f"Matches with deleted concepts: {deleted_matches[0]}")
    for key, val in deleted_matches[1].items():
        if val != []:
            print(key, val, sep = "\t")
    print("\n")

    # check identified substructures against present concepts
    present = load_json(args.present_concepts)
    present_matches = check_all_substructures(top_substructures, present)
    print(f"Matches with present concepts: {present_matches[0]}")
    for key, val in present_matches[1].items():
        if val != []:
            print(key, val, sep = "\t")
    print("\n")

    # for each deleted concept check if it was identified somewhere
    discovered = check_deleted_against_identified(deleted, deleted_matches[1])
    print(f"Deleted concepts that were discovered: {discovered[0]}")
    for key, val in discovered[1].items():
        print(key, val, sep = "\t")

def main_seq(args):

    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)

    # read SubgraphX results
    df = pd.read_csv(args.subgraphx_file, header = 0)
    print(f"Eval for {args.subgraphx_file} \n")

    for i in range(1, 7):
        print(f"max_nodes = {i}")

        # get cooccurrence counts with positive class for every substructure
        cooc_counts = get_counts(df, i = i)
        top_substructures = [idx for idx in cooc_counts.index if "," not in idx]
        print(f"Substructures by cooccurrence count:\n{cooc_counts}\n")

        # get drawings of identified substructures
        outfolder_drawings = os.path.join(args.outfolder, "substructure_drawings")
        if not os.path.exists(outfolder_drawings):
            os.makedirs(outfolder_drawings)
        visualise_substructures(cooc_counts, outfolder = outfolder_drawings, min_count = 2)

        # get positive molecules with all positive concepts deleted
        special_molecules = get_special(df, i_max = 6)
        print(f"Special molecules:\n{special_molecules}\n")

        # for special molecules, draw molecule with highlighted substructure
        outfolder_special = os.path.join(args.outfolder, "special_highlights")
        if not os.path.exists(outfolder_special):
            os.makedirs(outfolder_special)
        draw_highlighted_molecules(special_molecules, outfolder = outfolder_special, i = i)

        # check identified substructures against deleted concepts
        deleted = load_json(args.deleted_concepts)
        deleted_matches = check_all_substructures(top_substructures, deleted, use_qq = False)
        print(f"Matches with deleted concepts: {deleted_matches[0]}")
        for key, val in deleted_matches[1].items():
            if val != []:
                print(key, val, sep = "\t")
        print("\n")

        # check identified substructures against present concepts
        present = load_json(args.present_concepts)
        present_matches = check_all_substructures(top_substructures, present)
        print(f"Matches with present concepts: {present_matches[0]}")
        for key, val in present_matches[1].items():
            if val != []:
                print(key, val, sep = "\t")
        print("\n")

        # for each deleted concept check if it was identified somewhere
        discovered = check_deleted_against_identified(deleted, deleted_matches[1])
        print(f"Deleted concepts that were discovered: {discovered[0]}")
        for key, val in discovered[1].items():
            print(key, val, sep = "\t")

        print("\n------------------------------\n")


if __name__ == "__main__":
    present_default = "data/mutag_dataset_remove_nitro/substructures.json"
    deleted_default = "data/mutag_dataset_remove_nitro/substructures_removed.json"
    perfect_default = "data/mutag_perfect/test_mutag.csv"
    out_default = "gnnexplainer_eval"

    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraphx-file", type = str, help = "path to csv file with identified substructures")
    ap.add_argument("--present-concepts", type = str, nargs = "?", default = present_default, help = "path to json file with present concepts")
    ap.add_argument("--deleted-concepts", type = str, nargs = "?", default = deleted_default, help = "path to json file with deleted concepts")
    ap.add_argument("--perfect-file", type = str, nargs = "?", default = perfect_default, help = "path to csv file with perfect dataset (all concepts)")
    ap.add_argument("--outfolder", type = str, nargs = "?", default = out_default, help = "path to directory where eval is written to")
    ap.add_argument("--seqfile", action = "store_true", help = "switch on eval for multiple columns")
    args = ap.parse_args()

    if args.seqfile:
        main_seq(args)
    else:
        main(args)