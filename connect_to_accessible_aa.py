import cupy as cp

def connect_to_accessible_aa(pos, sel, n_vectors=5000, step=0.1, initial_len=1.5, max_len=4, device_idx=0):

    '''
    Inputs: pos = Coordinate Matrix of the substrate atoms
            sel = Coordinate Matrix of the protein atoms

    Initialize n random vectors (n_vectors) around each atom of the substrate molecule (like a star). Iteratively increase the length 
    of the vectors. If a vector enters a 1A-sphere around an enzyme atom, this atoms (or rather the amino acid it belongs to)
    is selected as a neighbor of the substrate atom the vector belongs to. Vectors that have found an atom are no longer iteratively 
    elongated. This should ensure that only surface-accessible atoms and amino acids are included in the interaction graphs.'''

    n_iter = int((max_len-initial_len)/step) # starlines can only grow until n_iter * step (threshold for closeness)
    n_stars = pos.shape[0]

    with cp.cuda.Device(device_idx): #Move the arrays to the GPU with CuPy

        # Access default memory pool instance
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        pos_gpu = cp.array(pos)
        sel_gpu = cp.array(sel)

        # Initialize origins and random directions
        stars = pos_gpu[:, cp.newaxis, :]
        random_vectors = cp.random.uniform(size=(n_stars, n_vectors, 3), high=1.0, low=-1.0)
        norms = cp.linalg.norm(random_vectors, axis=2)
        random_vectors = random_vectors / norms[:,:,cp.newaxis] * step

        # Combine the atomscoords of substrate and protein
        atomcoords_combined = cp.concatenate([sel_gpu, pos_gpu], axis=0)

        #Initialize tensor to save faound neighborhood information
        neighborhood = cp.full((n_stars, n_vectors, atomcoords_combined.shape[0]), False)

        # Initialize origins and random directions
        stars = pos_gpu[:, cp.newaxis, :]
        random_vectors = cp.random.uniform(size=(n_stars, n_vectors, 3), high=1.0, low=-1.0)
        norms = cp.linalg.norm(random_vectors, axis=2)
        random_vectors = random_vectors / norms[:,:,cp.newaxis] * step

        # Elongate starlines to initial_len
        stars = stars + (1.4/step) * random_vectors

        for _ in range(n_iter):
            # Propagate the starlines by the random directions
            stars = stars + random_vectors

            # Calculate the pairwise distances between the starline tips and the atoms of the protein
            diff = atomcoords_combined[cp.newaxis, :, :] - stars[:, :, cp.newaxis, :]
            pairwise_distances = cp.linalg.norm(diff, axis=3)

            # Check if any atom of the protein is located closely to a starline tip and add the corresponding
            # residue to the list of surface residues
            bool = pairwise_distances < 1
            neighborhood = cp.logical_or(neighborhood, bool)

            successful_lines = cp.any(bool, axis=2)

            random_vectors[successful_lines] = 0

        selected_atoms = cp.any(neighborhood, axis=1)

        # Free the GPU Memory
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return cp.asnumpy(selected_atoms)