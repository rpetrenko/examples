#!/usr/bin/env python
from __future__ import print_function
from itertools import count

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5 # Tensor size 4x1
b_target = torch.randn(1) * 5 # Tensor size 1


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    """Creates a string description of a polynomial.
    W - Tensor size 4
    b - Tensor size 1
    """
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1) # converts size 32 to 32x1
    poly = [x ** i for i in range(1, POLY_DEGREE+1)] # list of size 4 of Tensors 32x1
    y = torch.cat(poly, 1)
    return y # 32x4


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size) # Tensor size 32
    x = make_features(random) # Tensor size 32x4
    y = f(x) # Tensor size 32x1
    return Variable(x), Variable(y)


# Define model
# W_target size (4, 1)
fc = torch.nn.Linear(W_target.size(0), 1)

for batch_idx in count(1):
    # Get data
    # batch_x: Variable Tensor 32x4
    # batch_y: Variable Tensor 32x1
    x_target, y_target = get_batch()

    # Reset gradients
    fc.zero_grad()

    # Forward pass (compute y from x)
    y_pred = fc(x_target)
    output = F.smooth_l1_loss(y_pred, y_target) # Variable Tensor 1
    loss = output.data[0] # number

    # Backward pass (compute gradients)
    output.backward()

    # Apply gradients
    for param in fc.parameters():
        # learning rate
        learning_rate = 0.1
        param.data.add_(-learning_rate * param.grad.data)

    # Stop criterion
    if loss < 1e-2:
        break

    if batch_idx % 100 == 0 or batch_idx == 1:
        # reshape
        weights = fc.weight.data # Tensor size 1x4
        weights = weights.view(-1) # Tensor size 4
        biases = fc.bias.data # Tensor size 1
        func_str = poly_desc(weights, biases)
        print("Idx {:5d} Loss {:7.3f} function {}".format(batch_idx, loss, func_str))

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))

visualize = True
# if visualize:
#     pass