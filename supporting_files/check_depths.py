def check():
    nodes = {}
    with open('fort.14', 'r') as f:
        for i, line in enumerate(f):
            if i < 2: continue
            p = line.split()
            if len(p) >= 4:
                try:
                    nodes[int(p[0])] = float(p[3])
                except: pass
            if i > 70000: break
    n_adj = [13555, 12836, 12126, 12125, 12127, 12128, 12834, 13554]
    print('Node 12835:', nodes[12835])
    for n in n_adj: print(f'Node {n}: {nodes[n]}')

if __name__ == '__main__':
    check()
