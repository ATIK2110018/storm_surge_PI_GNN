import math

def check_mesh():
    with open('fort.14', 'r') as f:
        lines = f.readlines()
    
    parts = lines[1].split()
    n_elements = int(parts[0])
    n_nodes = int(parts[1])
    
    nodes = {}
    for i in range(2, 2 + n_nodes):
        p = lines[i].split()
        nid = int(p[0])
        x = float(p[1])
        y = float(p[2])
        nodes[nid] = (x, y)
        
    bad_elems = []
    min_area = float('inf')
    max_area = float('-inf')
    
    for i in range(2 + n_nodes, 2 + n_nodes + n_elements):
        p = lines[i].split()
        eid = int(p[0])
        n1 = int(p[2])
        n2 = int(p[3])
        n3 = int(p[4])
        
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x3, y3 = nodes[n3]
        
        # Signed area (2x)
        area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        
        min_area = min(min_area, area)
        max_area = max(max_area, area)
        
        if area <= 0:
            bad_elems.append((eid, area))
            
    print(f"Checked {n_elements} elements.")
    print(f"Min area (x2): {min_area}, Max area (x2): {max_area}")
    print(f"Found {len(bad_elems)} elements with <= 0 area.")
    if bad_elems:
        print("First 10 bad elems:", bad_elems[:10])

if __name__ == '__main__':
    check_mesh()
