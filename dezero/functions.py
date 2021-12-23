import numpy as np

from dezero import utils
from dezero.core import Function, as_variable


class Sin(Function):
    def forward(self, xs):
        y = np.sin(xs)
        return y

    def backward(self, gys):
        x, = self.inputs
        gx = gys * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, xs):
        y = np.cos(xs)
        return y

    def backward(self, gys):
        x, = self.inputs
        gx = gys * - sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, xs):
        y = np.tanh(xs)
        return y

    def backward(self, gys):
        y = self.outputs[0]()
        gx = gys * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, xs):
        self.x_shape = xs.shape
        y = xs.reshape(self.shape)
        return y

    def backward(self, gys):
        return reshape(gys, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, xs):
        y = np.transpose(xs)
        return y

    def backward(self, gys):
        gx = transpose(gys)
        return gx


def transpose(x):
    return Transpose()(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        # keepdims은 입력과 풀력의 차원수를 똑같게 유지할지 정함
        # False일시 y의 형상은 (), 즉 스칼라
        # True일시 축의 수가 유지
        self.keepdims = keepdims

    def forward(self, xs):
        self.x_shape = xs.shape
        y = xs.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gys):
        # reshape_sum_backward 는 gy의 형상을 미세하게 조정
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gys, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, xs):
        self.x_shape = xs.shape
        y = np.broadcast_to(xs, self.x_shape)
        return y

    def backward(self, gys):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, xs):
        self.x_shape = xs.shape
        y = utils.sum_to(xs, self.shape)
        return y

    def backward(self, gys):
        gx = broadcast_to(gys, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


# 행렬 곱
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)
