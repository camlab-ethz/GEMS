import os
import argparse
from Bio import PDB

# Set up argument parser
parser = argparse.ArgumentParser(description="Process a PDB file to remove water molecules.")
parser.add_argument("path", type=str, help="Path to the PDB file")

# Parse arguments
args = parser.parse_args()
id = os.path.basename(args.path)
print(id)

# Initialize parser and structure using the path argument
pdb_parser = PDB.PDBParser()
structure = pdb_parser.get_structure(id, args.path)

# Initialize the Select class to ignore water molecules
class NoWaters(PDB.Select):
    def accept_residue(self, residue):
        if residue.get_resname() == "HOH":
            return 0
        else:
            return 1

# Initialize IO
io = PDB.PDBIO()
io.set_structure(structure)

# Save the structure without waters
# Assuming you want to save the output in the same directory but with a modified file name
output_path = args.path.rsplit(".", 1)[0] + "_nowaterr.pdb"
io.save(output_path, NoWaters())

print(f"Processed file saved to: {output_path}")
