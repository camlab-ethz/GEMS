import numpy as np

def calculate_cbeta_position(ca_coords, c_coords, n_coords):
    # Convert input coordinates to numpy arrays
    ca = np.array(ca_coords)
    c = np.array(c_coords)
    n = np.array(n_coords)
    
    # Bond lengths and angles
    bond_length_ca_cb = 1.54  # Å
    bond_angle_n_ca_cb = np.deg2rad(109.5)  # radians
    bond_angle_c_ca_cb = np.deg2rad(109.5)  # radians
    
    # Unit vectors along the bonds
    u_n_ca = (n - ca) / np.linalg.norm(n - ca)
    u_c_ca = (c - ca) / np.linalg.norm(c - ca)
    
    # Orthogonal vector to the plane formed by N, Cα, and C
    u_orth = np.cross(u_n_ca, u_c_ca)
    u_orth /= np.linalg.norm(u_orth)  # Normalize
    
    # Vector component in the plane
    u_plane = np.cross(u_orth, u_n_ca)
    u_plane /= np.linalg.norm(u_plane)  # Normalize
    
    # Compute the Cβ position
    cb = (ca + bond_length_ca_cb * (
        np.cos(bond_angle_n_ca_cb) * u_n_ca +
        np.sin(bond_angle_n_ca_cb) * (
            np.cos(bond_angle_c_ca_cb) * u_plane +
            np.sin(bond_angle_c_ca_cb) * u_orth)
    ))
    
    return cb

# Example usage:
ca_coords = [1.0, 1.0, 1.0]
c_coords = [2.0, 1.5, 1.5]
n_coords = [1.5, 2.0, 1.0]

cb_coords = calculate_cbeta_position(ca_coords, c_coords, n_coords)
print("Cβ coordinates:", cb_coords)