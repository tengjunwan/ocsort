import numpy as np
from scipy.optimize import linear_sum_assignment

from hungarian_algorithm import hungarian_rect


# def linear_assignment(cost_matrix):
#     try:
#         import lap
#         print("import lap")
#         _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
#         return np.array([[y[i],i] for i in x if i >= 0]) #
#     except ImportError:
#         print("import scipy")
#         from scipy.optimize import linear_sum_assignment
#         x, y = linear_sum_assignment(cost_matrix)
#         print(f"x: {x}")
#         print(f"y: {y}")
#         return np.array(list(zip(x, y)))
    

def linear_assignment_scipy(cost_matrix):

    
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))
    

if __name__ == "__main__":
    # cost_matrix = np.array([[1, 0, 1], 
    #                         [1, 1, 0],
    #                         [0, 1, 1]], dtype=np.float32)

    # matching_scipy = linear_assignment_scipy(cost_matrix)
    # cost_scipy = 0
    # for m in matching_scipy:
    #     cost_scipy += cost_matrix[m[0], m[1]]
    # cost_my, matching_my = hungarian_rect(cost_matrix)
    # if cost_my != cost_scipy:
    #     print(cost_scipy, matching_scipy)
    #     print(cost_my, matching_my)


    def random_array(h: int, w: int, dtype=np.int64) -> np.ndarray:
        if dtype == np.int64:
            return np.random.randint(0, 100, size=(h, w), dtype=dtype)
        elif dtype == np.float64:
            return np.random.rand(h, w).astype(dtype) * 100
        else:
            raise ValueError("Only np.int64 and np.float64 are supported.")


    cost_matrices = [random_array(3, 6, dtype=np.float64) for i in range(10000)]
    for cost_matrix in cost_matrices:
        matching_scipy = linear_assignment_scipy(cost_matrix)
        cost_scipy = 0
        for m in matching_scipy:
            cost_scipy += cost_matrix[m[0], m[1]]
        cost_my, matching_my = hungarian_rect(cost_matrix)
        if cost_my != cost_scipy:
            print("negavite")
            print(cost_scipy, matching_scipy)
            print(cost_my, matching_my)
            print(cost_matrix)
            raise ValueError("something wrong")
        else:
            print(f'positive: {cost_scipy} == {cost_my}')


# a = np.array([[52.65841514, 61.10300038, 31.29519318, 79.38503696, 79.64286437, 73.56810781],
#               [31.38967455,  4.40757004, 42.6980129,   8.15217604, 54.0112521,  98.88227511],
#               [74.11460562, 50.40560778, 61.8812237,  43.3962529,  43.30836459, 67.81267483]], np.float64)

# b= np.array([[37.31153277, 19.05192666, 69.30708344, 82.6275153,  22.51775091, 79.80133828],
#              [99.72771733,  2.80065632, 68.88984273, 70.69600235, 66.186798,    1.16486698],
#              [80.10191983, 35.86268588,  5.69608423, 63.65792089, 10.27934824,  0.38223899]], dtype=np.float64)
# matching_scipy = linear_assignment_scipy(b)
# cost_scipy = 0
# for m in matching_scipy:
#     cost_scipy += b[m[0], m[1]]
# cost_my, matching_my = hungarian_rect(b)
# if cost_my != cost_scipy:
#     print(cost_scipy, matching_scipy)
#     print(cost_my, matching_my)