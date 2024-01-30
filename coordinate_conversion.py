import sys
import warnings
import argparse

import numpy as np
from numpy import linalg as LA
from qcelemental import constants
from qcelemental.models import Molecule
from qcelemental.molutil import guess_connectivity

# ====================================
# Helper functions for builder objects
# ====================================

# Functions for converting molecular coordinates between xyz and zmat format

# Build 3D matrix for rotation about axis n
#   theta: degrees of rotaion about n and n given as [x,y,z]
#   n: the axis of rotation
#   degrees: True if theta is in degrees (default: True)
def build_rotation_matrix(theta, n, degrees=True):
    if degrees:
        theta *= np.pi/180    
    c = np.cos(theta)
    s = np.sin(theta)
    x, y, z = n / LA.norm(n)

    rmat = np.array([
        [c + x**2*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [x*y*(1-c) + z*s, c + y**2*(1-c), y*z*(1-c) - x*s],
        [x*z*(1-c) - y*s, y*z*(1-c) + x*s, c + z**2*(1-c)]
    ])
    
    return rmat

# Rotate a vector by theta degrees about axis n
#   vector: the vector being rotation
#   theta: degrees of rotation
#   n: the axis of rotation
#   degrees: True if theta is in degrees (default: True)
def rotate_vector(vector, theta, n, degrees=True):
    rmat = build_rotation_matrix(theta, n, degrees=degrees)
    new_vector = np.dot(rmat, vector)
    return new_vector

# Calculate a new point from zmat specifications
#   values: list of values for the zmat specifications, i.e. [R, A, D]
#   geometry: list of cartesian coordinates for the indices in the zmat
def calculate_position(values, geometry):
    R, A, D = values
    p1, p2, p3 = geometry
    r_1_2 = p2 - p1                    # bond vector for atoms 1 and 2
    p4 = r_1_2 * R / LA.norm(r_1_2)    # new bond of length R at 0 degrees
    n = np.cross(r_1_2, p3 - p1)       # rotation axis
    p4 = rotate_vector(p4, A, n)
    n = -r_1_2                         # dihedral rotation axis
    p4 = rotate_vector(p4, D, n)
    return p4 + p1

# Calculate the distance between 2 points
def get_distance(p1, p2):
    return LA.norm(p1-p2)

# Calculate the angle between 3 points as p1-p2-p3
def get_angle(p1, p2, p3):
    u1 = p1 - p2
    u1 /= LA.norm(u1)
    u2 = p3 - p2
    u2 /= LA.norm(u2)
    return np.arccos(np.dot(u1, u2)) * 180 / np.pi

# Calculate the dihedral between 4 points as p1-p2-p3-p4
def get_dihedral(p1, p2, p3, p4):
    u1, u2, u3 = p1 - p2, p3 - p2, p4 - p3
    u1 /= LA.norm(u1)
    u2 /= LA.norm(u2)
    u3 /= LA.norm(u3)
    n1, n2 = np.cross(u1, u2), np.cross(-u2, u3)
    m = np.cross(-u2, n2)
    x = np.dot(n1, n2)
    y = np.dot(n1, m)
    return np.arctan2(y, x) * 180 / np.pi

# Warn users if the ZMAT builder can't find a bond
def no_bond_found(atom_index, coordinate):
    warnings.warn(f"No connecting bond found for {coordinate} coordinate of "
				  f"atom index {atom_index + 1}. Using nearest atom instead.")
    
# Find the closest atom which hasn't been used
def get_nearest_atom(atom_ind, geometry, connection_ind):
    ind = 0        # Placeholder for index of nearest atom
    min_dist = 100 # large initial value
    for i in range(atom_ind):
        # If it is already connected, skip it
        if i in connection_ind:
            continue
        # Keep the index with the smallest distance
        dist = LA.norm(geometry[i] - geometry[atom_ind])
        if dist < min_dist:
            min_dist = dist
            ind = i
            
    return ind

angstrom2bohr = constants.conversion_factor('angstrom', 'bohr')


# ===============================================================
# Builder objects for converting between xyz and zmat coordinates
# ===============================================================

class CartesianBuilder:
    # get zmat information from string
    def __init__(self, zmat_str):
        self.symbols = []   # list of atoms in molecule
        self.indices = []     # indicies from zmat for connectivity
        self.values = {}      # values for zmat connectivity
        lines = zmat_str.splitlines()
        self.title = lines[0]
        for line in lines[1:]:
            items = line.split()
            
            # Skip empty lines
            if items == []:
                continue
            # Add variable assignments to values dict
            elif "=" in items:
                self.values[items[0]] = float(items[2])
            # Read in ZMAT specification
            else:
                self.symbols.append(items[0])
                self.indices.append([[int(i) for i in items[1::2]], items[2::2]])    
                
    # Calculate cartesian coordinate output in bohr
    def calculate_cartesian(self):
        natoms = len(self.symbols)

        # First atom on the origin
        geometry = np.zeros((natoms, 3))
        
        # Second atom on x-axis
        R = self.indices[1][1][0]
        
        # If the ZMAT uses variables, get the value from the dict
        if type(R) == str:
            geometry[1,0] = self.values[R]
        # Otherwise, use the value that's there
        else:
            geometry[1,0] = R
        
        # Third atom in xy-plane
        R, A = self.indices[2][1]  
        
        # Get values from dict if needed
        if type(R) == str:
            R = self.values[R]
        if type(A) == str:
            A = self.values[A]
        
        # Rotated vector
        geometry[2,0] = R * np.cos(A * np.pi/180)
        geometry[2,1] = R * np.sin(A * np.pi/180)
        
        # If connected to atom 2, subtract from its position
        if self.indices[2][0][0] == 2:
            geometry[2] = geometry[1] - geometry[2]
        
        # Build the rest of the molecule from these coordinates
        for i in range(3, natoms):
            ind = [x - 1 for x in self.indices[i][0]]
            vals = self.indices[i][1]
            for j in range(3):
                if type(vals[j]) == str:
                    vals[j] = self.values[vals[j]]
            geometry[i] = calculate_position(vals, geometry[ind])
            
        # We use QCElemental to orient the molecule
        mol = Molecule(symbols=self.symbols, geometry=geometry, orient=True)
        
        return mol.geometry
    
    # output an xyz string
    def build_string(self, comment=None):
        natoms = len(self.symbols)
        geometry = self.calculate_cartesian()
        if not comment:
            comment = self.title
        out = f"{natoms}\n{comment}\n"
        for i in range(natoms):
            out += (f"{self.symbols[i]:<4}  {geometry[i,0]:>14.10f}  "
					f"{geometry[i,1]:>14.10f}  {geometry[i,2]:>14.10f}\n")
        return out

class ZMATBuilder:
    # get structure information from string
    def __init__(self, cart_str, units='angstrom'):
        lines = cart_str.splitlines()
        natoms = int(lines[0])
        self.symbols = np.zeros(natoms, dtype=object)
        self.geometry = np.zeros((natoms, 3))
        self.title = lines[1] # Comment line
        for i in range(natoms):
            items = lines[i+2].split()
            self.symbols[i] = items[0]
            self.geometry[i] = np.array([float(item) for item in items[1:]])
        self.units = units
                
    # Build the zmat from cartesian coordinates
    def build_zmat(self):
        symbols = self.symbols
        geometry = self.geometry
        if self.units == 'angstrom':
            connectivity = guess_connectivity(symbols, geometry * angstrom2bohr)
        else:
            connectivity = guess_connectivity(symbols, geometry)
        print(connectivity)
        variables = {}
        
        # Initialize list for zmat
        # First atom by itself, second atom at distance R1
        zmat = [
            [symbols[0]], 
            [symbols[1], 1, 'R1'],
        ]
        variables['R1'] = get_distance(geometry[0], geometry[1])
        
        # Third atom
        ind = [2, 1]
        # If bonded to atom 1 instead of atom 2, swap indices
        if (0, 2) == connectivity[1]:
            ind = [1, 2]
        zmat.append([symbols[2], ind[0], 'R2', ind[1], 'A1'])
        variables['R2'] = get_distance(geometry[2], geometry[ind[0]-1])
        variables['A1'] = get_angle(geometry[2],
                                    geometry[ind[0]-1], 
									geometry[ind[1]-1])
        
        # Fill out the rest of the lines
        for i in range(3,len(symbols)):
            ind = np.full(3, -1, dtype=int) # placeholders
            
            # Find the index of the first atom its bonded to
            for pair in connectivity:
                # Must be bonded to atom with lower index
                if i == pair[1]:
                    ind[0] = pair[0]
                    break
            
            # If no bond found, get the closest atom
            if ind[0] < 0:
                ind[0] = get_nearest_atom(i, geometry, ind)
                no_bond_found(i, "distance")
            
            # Get index of the next atom in chain with the lowest index
            # Must be bonded to an atom with a lower index than i
            if ind[0] == 0:
                ind[1] = connectivity[0][1]
            else:
                for pair in connectivity:
                    if ind[0] == pair[1]:
                        ind[1] = pair[0]
                        break
                        
            # If no bond found, get the closest atom
            if ind[1] < 0:
                ind[1] = get_nearest_atom(i, geometry, ind)
                no_bond_found(i, "angle")
              
            # Get index for the last atom in the chain with the lowest index
            # Must be lower index than i, can't already be in ind
            for pair in connectivity:
                # Atom i shouldn't be able to see atoms with higher indices
                # Only have to check pair[1] since it will always be largest
                if pair[1] >= i:
                    continue
                # Bond with ind[1] that is not the same as the bond with ind[0]
                if ind[1] == pair[1] and pair[0] != ind[0]:
                    ind[2] = pair[0]
                    break
                if ind[1] == pair[0] and pair[1] != ind[0]:
                    ind[2] = pair[1]
                    break
               
            # If no bond found, get the closest atom
            if ind[2] < 0:
                ind[2] = get_nearest_atom(i, geometry, ind)
                no_bond_found(i, "torsion")
            
            Rlabel, Alabel, Dlabel = f"R{i}", f"A{i-1}", f"D{i-2}"
            variables[Rlabel] = get_distance(geometry[i],
                                             geometry[ind[0]])
            variables[Alabel] = get_angle(geometry[i],
                                          geometry[ind[0]],
										  geometry[ind[1]])
            variables[Dlabel] = get_dihedral(geometry[i],
                                             geometry[ind[0]],
											 geometry[ind[1]],
											 geometry[ind[2]])
                        
            zmat.append([symbols[i], ind[0]+1, Rlabel, ind[1]+1, Alabel, 
						 ind[2]+1, Dlabel])
            
        # Add in variable assignments
        zmat.append([]) # Blank line
        for key in variables:
            zmat.append([key, "=", f"{variables[key]:>15.10f}"])
            
        return zmat

    # Build the ZMAT string
    def build_string(self, title=None):
        if not title:
            title = self.title
        zmat = self.build_zmat()
        out = title + "\n"
        for line in zmat:
            line = [str(item) for item in line]
            out += " ".join(line) + "\n"
            
        return out


# Main script contents
if __name__ == '__main__':

    # Parse arguments to find input file and format type
    parser = argparse.ArgumentParser(
    	description='Convert between xyz and ZMAT coordinates.'
    )
    parser.add_argument('filename',
    					help='filename for input file')
    parser.add_argument('format', choices=['zmat', 'xyz'],
    					help='format type of the input file')
    parser.add_argument('-o', '--output', nargs='?', const='output',
						help='store output to this filename')
    parser.add_argument('-u', '--units', choices=['angstrom', 'bohr'],
                        default='angstrom',
                        help='units of cartesian coordinates')
    args = parser.parse_args()

	# Open file and save contents as a string
    input_string = ''
    with open(args.filename, 'r') as f:
        input_string = f.read()

    # Build new string from input file
    builder = None
    if args.format == 'zmat':
        builder = CartesianBuilder(input_string)
    else:
        builder = ZMATBuilder(input_string, units=args.units)
    output_string = builder.build_string()

    # Output results
    if type(args.output) == str:
        with open(args.output, 'w') as f:
            f.write(output_string)
    else:
        sys.stdout.write(output_string)
