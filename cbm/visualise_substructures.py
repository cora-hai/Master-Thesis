import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display

def draw_highlighted_molecule_svg(smiles, smarts, width=500, height=400, filename=None):
    # 1. Parse molecule and substructure query
    mol = Chem.MolFromSmiles(smiles)
    query = Chem.MolFromSmarts(smarts)
    
    # Generate 2D coordinates if not present
    if not mol.GetNumConformers():
        rdMolDraw2D.PrepareMolForDrawing(mol)
        
    # 2. Find matching atoms and bonds for the substructure
    matches = mol.GetSubstructMatch(query)
    
    # highlight_atoms = []
    # highlight_bonds = []
    
    # if matches:
    #     # Collect all atom indices from matches
    #     match_atoms = set()
    #     for match in matches:
    #         match_atoms.update(match)
    #     highlight_atoms = list(match_atoms)
        
    #     # Collect all bond indices connecting the matched atoms
    #     for bond in mol.GetBonds():
    #         if bond.GetBeginAtomIdx() in match_atoms and bond.GetEndAtomIdx() in match_atoms:
    #             highlight_bonds.append(bond.GetIdx())

    # 3. Initialize the SVG drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    
    # 4. Optional: Fine-tune drawing options for publication quality
    dopts = drawer.drawOptions()
    dopts.bondLineWidth = 2.0         # Slightly thicker lines for clarity
    dopts.addAtomIndices = False      # Hide atom indices
    dopts.clearBackground = False     # Transparent background (useful for papers/slides)

    # 5. Draw molecule with highlights
    drawer.DrawMolecule(
        mol,
        highlightAtoms=matches,
        #highlightBonds=highlight_bonds
    )
    drawer.FinishDrawing()
    
    svg_text = drawer.GetDrawingText()
    
    # 6. Save to file if a filename is provided
    if filename:
        with open(filename, 'w') as f:
            f.write(svg_text)
            
    return svg_text


def visualise(filepath: str) -> None:

    # 1. Load your CSV file
    # Replace 'your_data.csv' with the actual path to your file
    df = pd.read_csv(filepath)

    # Create a directory to save the generated images
    output_dir = "molecule_substructure_images_nitro_new"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        drug_smiles = row["Drug"]
        substruct_smiles = row["Top Substructure"]
        drug_id = row["Drug_ID"]

        if "Disconnected" in substruct_smiles or row["Y"] == 0:
            continue

        # # Convert SMILES strings to RDKit molecule objects
        # mol = Chem.MolFromSmiles(drug_smiles)
        # substruct = Chem.MolFromSmarts(substruct_smiles)

        # # Check if SMILES are valid
        # if mol is None:
        #     print(f"Skipping row {index}: Molecule has invalid SMILES.")
        #     continue
        # elif substruct is None:
        #     print(f"Skipping row {index}: Substructure has invalid SMILES.")
        #     continue

        # # Set the atom label to 'C' for all carbon atoms
        # for atom in mol.GetAtoms():
        #     if atom.GetSymbol() == 'C':
        #         atom.SetProp('atomLabel', 'C')

        # # Set the atom label to 'C' for all carbon atoms
        # for atom in substruct.GetAtoms():
        #     if atom.GetSymbol() == 'C':
        #         atom.SetProp('atomLabel', 'C')

        # # Find atom indices where the substructure matches the molecule (for highlighting)
        # match_indices = mol.GetSubstructMatch(substruct)

        # 3. Generate side-by-side image
        # molsPerRow=2 creates a 1-row, 2-column grid containing [Molecule, Substructure]
        # img = Draw.MolsToGridImage(
        #     [mol, substruct],
        #     molsPerRow=2,
        #     subImgSize=(300, 300),
        #     legends=[f"Molecule:\n\n{drug_smiles}", f"Top Substructure:\n\n{substruct_smiles}"],
        #     highlightAtomLists=[match_indices, []],
        # )

        # img = Draw.MolDraw2DSVG(
        #         mol,
        #         subImgSize=(300, 300),
        #         legend = f"Molecule:\n\n{drug_smiles}",
        #         highlightAtomLists = match_indices,
        #     )

        # # 4. Save the image to the output directory
        # image_path = os.path.join(output_dir, f"{index}_{drug_id}.png")
        # img.save(image_path)

        svg_content = draw_highlighted_molecule_svg(
            smiles=drug_smiles,
            smarts=substruct_smiles,
            width=600, 
            height=450,
            filename=f"{output_dir}/{index}_{drug_id}.svg"
        )






if __name__ == "__main__":
    attribution_file = "feature_attribution_out/test_mutag_subgraphx_feat-attr_nitro_new.csv"
    visualise(attribution_file)