import sys

def fix_depths(fort14_path):
    with open(fort14_path, 'r') as f:
        lines = f.readlines()
        
    header = lines[1].strip().split()
    ne = int(header[0])
    nn = int(header[1])
    
    # 1. Identify open boundary nodes
    idx = 2 + nn + ne
    num_ob = int(lines[idx].split()[0])
    idx += 2
    
    ob_nodes = set()
    for _ in range(num_ob):
        n_nodes = int(lines[idx].split()[0])
        idx += 1
        for _ in range(n_nodes):
            ob_nodes.add(int(lines[idx].strip()))
            idx += 1
            
    # 2. Modify depths in nodes section
    modified = 0
    for i in range(2, 2 + nn):
        parts = lines[i].split()
        nid = int(parts[0])
        if nid in ob_nodes:
            depth = float(parts[3])
            if depth < 5.0:
                # Replace depth while preserving formatting as much as possible
                # standard OceanMesh format: 1    98.7369255079    12.0720488941   1.0000000000e+00
                new_line = f" {nid:9d} {float(parts[1]):16.10f} {float(parts[2]):16.10f}   5.0000000000e+00\n"
                lines[i] = new_line
                modified += 1
                print(f"Deepened Node {nid} from {depth} to 5.0")
                
    # 3. Write back
    with open(fort14_path, 'w') as f:
        f.writelines(lines)
        
    print(f"Total open boundary nodes deepened: {modified}")

if __name__ == "__main__":
    fix_depths('fort.14')
