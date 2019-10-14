import numpy as np
import random

fa = []
def get_fa(i):
    global fa
    if fa[i] != i:
        fa[i] = get_fa(fa[i])
    return fa[i]

def connectivity_check(n, edges):
    if n == 0:
        return 0
    global fa
    fa = [i for i in range(n)]
    for x, y in edges:
        x, y = get_fa(x), get_fa(y)
        if x != y:
            fa[x] = y
    fa_set = set()
    for i in range(n):
        fa_set.add(get_fa(i))
    return len(set(fa_set)) == 1

def sample_parser(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        tmp = f.read().split('\n')
        n, m = [int(x) for x in tmp[0].split(' ')]
        edges = []
        mat = np.zeros([n, n])
        for line in tmp[1:]:
            if line == '':
                continue
            x, y = [int(x) for x in line.split(' ')]
            edges.append([x, y])
            mat[x, y] = mat[y, x] = 1
        names = ['Concept_{}'.format(i) for i in range(n)]
    return names, edges, mat

def random_generator(n, m, output_file):
    assert n*(n-1)/2 >= m and n <= m
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append([i, j])
    while True:
        random.shuffle(edges)
        if connectivity_check(n, edges[:m]):
            break
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('{} {}\n'.format(n, m))
        for x, y in edges[:m]:
            f.write('{} {}\n'.format(x, y))

if __name__ == '__main__':
    print(connectivity_check(5, [[0,1],[1,2],[2,3],[3,4],[1,4]]))
    print(connectivity_check(5, [[0,1],[1,2],[2,3],[0,3],[1,3]]))