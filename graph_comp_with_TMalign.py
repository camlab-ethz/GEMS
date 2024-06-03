def write_pointcloud_to_pdb(points, output_pdb):
    with open(output_pdb, 'w') as pdb_file:
        for i, (x, y, z) in enumerate(points):
            pdb_file.write(
                "ATOM  {:5d}  CA  ALA A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C\n".format(
                    i + 1, 1, x, y, z  # All points belong to chain A (chain 1)
                )
            )
        pdb_file.write("TER\nEND\n")  # End of the chain and file

# Example point clouds
pointcloud1 = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]

pointcloud2 = [
    [1.1, 2.1, 3.1],
    [4.1, 5.1, 6.1],
    [7.1, 8.1, 9.1]
]

# Write point clouds to PDB files
write_pointcloud_to_pdb(pointcloud1, "/cluster/work/math/dagraber/DTI/TM-align/pointcloud_basic1.pdb")
write_pointcloud_to_pdb(pointcloud2, "/cluster/work/math/dagraber/DTI/TM-align/pointcloud_basic2.pdb")

import subprocess
import os

def run_tmalign(pdb1, pdb2, output_prefix="aligned"):
    tmalign_path = "/cluster/work/math/dagraber/DTI/TM-align/TMalign"
    
    # Check if TM-align executable exists
    if not os.path.isfile(tmalign_path):
        print(f"Error: TM-align executable not found at {tmalign_path}")
        return None
    
    # Check if the PDB files exist
    if not os.path.isfile(pdb1):
        print(f"Error: PDB file not found: {pdb1}")
        return None
    if not os.path.isfile(pdb2):
        print(f"Error: PDB file not found: {pdb2}")
        return None

    output_option = f"-o {output_prefix}"
    
    # Run TM-align with the two PDB files and output option
    process = subprocess.Popen([tmalign_path, pdb1, pdb2, output_option], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error running TM-align: {stderr.decode()}")
        return None

    # Parse the TM-score from the output
    output = stdout.decode()
    tm_score = None
    for line in output.split('\n'):
        if line.startswith("TM-score="):
            tm_score = float(line.split('=')[1].split()[0])
            break

    # Check for the output files
    aligned_pdb1 = f"{output_prefix}.pdb1"
    aligned_pdb2 = f"{output_prefix}.pdb2"
    if os.path.exists(aligned_pdb1) and os.path.exists(aligned_pdb2):
        print(f"Aligned PDB files generated: {aligned_pdb1}, {aligned_pdb2}")
    else:
        print("Aligned PDB files were not generated.")

    return tm_score

# Run TM-align on the point clouds
output_prefix = "aligned_pointclouds"
tm_score = run_tmalign("/cluster/work/math/dagraber/DTI/TM-align/pointcloud_basic1.pdb", "/cluster/work/math/dagraber/DTI/TM-align/pointcloud_basic2.pdb", output_prefix)
if tm_score is not None:
    print(f"TM-score: {tm_score}")