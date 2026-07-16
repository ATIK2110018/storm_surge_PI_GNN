import math

def check():
    nodes = {}
    with open('fort.14', 'r') as f:
        for i, line in enumerate(f):
            if i < 2: continue
            p = line.split()
            if len(p) >= 4:
                try:
                    nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
                except: pass
            if i > 70000: break

    n12835 = nodes[12835]
    n_adj = [13555, 12836, 12126, 12125, 11174, 11905, 11902, 12127, 12128, 12834, 13554]
    print(f"Node 12835 depth: {n12835[2]} m")
    for n in n_adj:
        dx = (nodes[n][0] - n12835[0]) * 111111 * math.cos(math.radians(n12835[1]))
        dy = (nodes[n][1] - n12835[1]) * 111111
        dist = math.hypot(dx, dy)
        cfl = (math.sqrt(9.81 * n12835[2]) * 2.0) / dist
        print(f"Dist to {n}: {dist:.2f} meters, CFL: {cfl:.2f}")

if __name__ == '__main__':
    check()
