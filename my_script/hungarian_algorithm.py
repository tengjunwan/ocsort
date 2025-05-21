import numpy as np


def hungarian(cost):
    J, W = cost.shape
    assert J == W
 
    # job[w] = job assigned to worker w (or -1 if unassigned)
    # We add one dummy worker at index W to simplify augmenting path logic
    job = np.full(W + 1, -1, dtype=int)
 
    # Dual variables (potentials) for jobs (ys) and workers (yt)
    ys = np.zeros(J, dtype=np.float64)
    yt = np.zeros(W + 1, dtype=np.float64)
 
    inf = np.iinfo(np.int64).max
 
    for j_cur in range(J):
        # Start a new alternating tree with dummy worker W as root
        w_cur = W
        job[w_cur] = j_cur
 
        # Slack values: min_to[w] = min slack from any node in tree to worker w
        min_to = np.full(W + 1, inf, dtype=np.float64)
 
        # prev[w] stores the previous worker on the alternating path to w
        prev = np.full(W + 1, -1, dtype=int)
 
        # in_z[w] indicates if worker w is already in the alternating tree Z
        in_z = np.zeros(W + 1, dtype=bool)
 
        while job[w_cur] != -1:
            in_z[w_cur] = True
            j = job[w_cur]
            delta = inf
            w_next = -1
 
            # Scan all candidate slack branches from current tree to workers
            for w in range(W):
                if in_z[w]:
                    continue
                rcost = cost[j, w] - ys[j] - yt[w]
                if rcost < min_to[w]:
                    min_to[w] = rcost
                    prev[w] = w_cur
                if min_to[w] < delta:
                    delta = min_to[w]
                    w_next = w
 
            # Update potentials to make at least one new edge tight
            for w in range(W + 1):
                if in_z[w]:
                    ys[job[w]] += delta
                    yt[w] -= delta
                else:
                    min_to[w] -= delta
 
            # Move to the next worker to grow the tree
            w_cur = w_next
 
        # Backtrack along alternating path and flip assignments
        while w_cur != W:
            job[w_cur] = job[prev[w_cur]]
            w_cur = prev[w_cur]
 
    # Build assignment: job[j] = assigned worker index
    # i.e. [worker_idx]
    assignment = [-1] * J
    for w in range(W):
        if job[w] != -1:
            assignment[job[w]] = w
 
    total_cost = sum(cost[j][assignment[j]] for j in range(J))

    return total_cost, assignment


def hungarian_rect(cost):
    J, W = cost.shape
    N = max(J, W)

    # Pad to square matrix with large constant
    pad_value = cost.max() + 1
    padded_cost = np.full((N, N), pad_value, dtype=cost.dtype)
    padded_cost[:J, :W] = cost

    # Call existing Hungarian algorithm on the padded matrix
    total_cost, assignment = hungarian(padded_cost)

    # Filter out dummy assignments
    matching = []
    filtered_cost = 0
    for j, w in enumerate(assignment):
        if j < J and w < W:
            matching.append((j, w))
            filtered_cost += cost[j][w]

    matching = np.array(matching, dtype=np.int64)
    return filtered_cost, matching


if __name__ == "__main__":
    cost = np.array([[1, 0, 1, 1],
                     [1, 1, 0, 1],
                     [0, 1, 1, 1]])
    
    print(hungarian_rect(cost))

    cost = np.array([[15, 6, 12, 8],
                     [10, 16, 8, 12],
                     [30, 25, 11, 9],
                     [13, 7, 20, 17],])
    print(hungarian_rect(cost))
    pass