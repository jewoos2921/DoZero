import numpy as np
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
