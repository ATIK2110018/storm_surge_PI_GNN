import sys

def generate_svg(fort14_path, svg_path):
    with open(fort14_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[1].strip().split()
    ne = int(header[0])
    nn = int(header[1])
    
    nodes = {}
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    # Read nodes
    for i in range(2, 2 + nn):
        parts = lines[i].strip().split()
        idx = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        nodes[idx] = (x, y)
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
        
    width = 800
    height = 800
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    if range_x == 0: range_x = 1
    if range_y == 0: range_y = 1
    
    scale = min(width / range_x, height / range_y) * 0.9
    
    def transform(x, y):
        cx = (x - min_x) * scale + (width - range_x * scale) / 2
        cy = height - ((y - min_y) * scale + (height - range_y * scale) / 2)
        return cx, cy

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background-color: white;">']
    
    # Draw all nodes as faint dots (mesh representation)
    svg.append('<g fill="#d3d3d3">')
    for x, y in nodes.values():
        cx, cy = transform(x, y)
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="0.5"/>')
    svg.append('</g>')

    # Skip elements
    idx = 2 + nn + ne
    
    # Open boundaries
    parts = lines[idx].strip().split()
    num_open_boundaries = int(parts[0])
    idx += 2 # skip total nodes line
    
    svg.append('<g stroke="blue" fill="none" stroke-width="2">')
    for _ in range(num_open_boundaries):
        parts = lines[idx].strip().split()
        num_nodes = int(parts[0])
        idx += 1
        points = []
        for _ in range(num_nodes):
            n_idx = int(lines[idx].strip())
            cx, cy = transform(*nodes[n_idx])
            points.append(f"{cx:.2f},{cy:.2f}")
            idx += 1
        svg.append(f'<polyline points="{" ".join(points)}"/>')
    svg.append('</g>')
    
    # Land boundaries
    parts = lines[idx].strip().split()
    num_land_boundaries = int(parts[0])
    idx += 2
    
    svg.append('<g stroke="green" fill="none" stroke-width="1.5">')
    for _ in range(num_land_boundaries):
        parts = lines[idx].strip().split()
        num_nodes = int(parts[0])
        idx += 1
        points = []
        for _ in range(num_nodes):
            n_idx = int(lines[idx].strip())
            cx, cy = transform(*nodes[n_idx])
            points.append(f"{cx:.2f},{cy:.2f}")
            idx += 1
        svg.append(f'<polyline points="{" ".join(points)}"/>')
    svg.append('</g>')

    # Legend
    svg.append('<rect x="10" y="10" width="180" height="70" fill="white" stroke="black" stroke-width="1"/>')
    svg.append('<line x1="20" y1="30" x2="40" y2="30" stroke="blue" stroke-width="3"/>')
    svg.append('<text x="50" y="35" font-family="Arial" font-size="14">Open Boundary (Tides)</text>')
    svg.append('<line x1="20" y1="60" x2="40" y2="60" stroke="green" stroke-width="2"/>')
    svg.append('<text x="50" y="65" font-family="Arial" font-size="14">Land Boundary</text>')
    
    svg.append('</svg>')
    
    with open(svg_path, 'w') as f:
        f.write("\n".join(svg))
    print(f"SVG saved to {svg_path}")

if __name__ == "__main__":
    generate_svg('fort.14', 'mesh_boundaries.svg')
