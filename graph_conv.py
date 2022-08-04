import numpy as np
# import scipy.sparse as sp
import torch
from torch.nn.functional import normalize

def calculate_laplacian_with_self_loop2(matrix):
    matrix = matrix + torch.eye(matrix.size(1)).unsqueeze(0).repeat(matrix.size(0), 1, 1).cuda()
    row_sum = matrix.sum(-1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt = d_inv_sqrt.unsqueeze(-1)
    d_mat_inv_sqrt = d_inv_sqrt*torch.eye(matrix.size(1)).unsqueeze(0).repeat(matrix.size(0), 1, 1).cuda()
    normalized_laplacian =torch.matmul(torch.matmul(d_mat_inv_sqrt,matrix).permute([0,2,1]),d_mat_inv_sqrt)
    return normalized_laplacian