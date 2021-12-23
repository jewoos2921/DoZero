import numpy as np
from dezero.core import Function


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
