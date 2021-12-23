import numpy as np
import weakref
import contextlib

"""
참조 카운트 방식의 메모리 관리
대입 연산자를 사용할 때
함수에 인수로 전달할 때
컨테이너 타입 객체(리스트, 튜플, 클래스 등)에 추가할 때

weakref 약한 참조
다른 객체를 참조하되 참조 카운트는 증가시키지 않는 기능
"""


class Variable:
    # 연산자 우선순위 - Variable 인스턴스의 연산자 우선순위를 ndarray 인스턴스의 연산자 우선순위 보다 높일 수 있음
    __array_priority__ = 200

    def __init__(self, data, name=None):
        # ndarray 만 취급하기
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은 (는) 지원하지 않습니다. ".format(type(data)))

        self.data = data
        self.name = name  # 변수 이름 지정
        self.grad = None  # 기울기
        self.creator = None
        self.generation = 0  # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.genertion + 1  # 세대수를 기록한다.(부모 세대 + 1)

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()  # funcs 리스트에 같은 함수를 중복 추가하는 이를 막기위해 사용

        def add_func(f_):
            if f_ not in seen_set:
                funcs.append(f_)
                seen_set.add(f_)
                funcs.sort(key=lambda x_: x_.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # 1. 출력 변수인 outputs에 담겨있는 미분값들을 리스트에 담는다.
            gxs = f.backward(*gys)  # 2. f 의 역전파를 호출한다.
            if not isinstance(gxs, tuple):  # 3. 튜플이 아니라면 튜플로 반환한다.
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):  # 4. 미분값을 변수 grad에 저장한다.
                if x.grad is None:
                    x.grad = gx  # 여기가 실수! 수정이 필요
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y는 약한 참조 실행시 참조 카운터가 0이 되어 미분값 데이터가 메모리에서 삭제

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    # def __mul__(self, other):
    #     return mul(self, other)


class Function(object):
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):  # 튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # 모드 전환
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs  # inputs은 역전파 계산시에만 사용 추론시에는 단순히 순전파만 사용해서 중간 계산 결과를 곧바로 버리면 메모리 사용량을 크게 줄임
            self.outputs = [weakref.ref(output) for output in outputs]

        #  리스트의 원소가 하나라면 첫 번째 원소를 반환한다.
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gys):
        return gys, gys


class Square(Function):
    def forward(self, xs):
        return xs ** 2

    def backward(self, gys):
        x = self.inputs[0].data
        gx = 2 * x * gys
        return gx


class Exp(Function):
    def forward(self, xs):
        return np.exp(xs)

    def backward(self, gys):
        x = self.inputs.data
        gx = np.exp(x) * gys
        return gx


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gys):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gys * x1, gys * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# 음수 부호
class Neg(Function):
    def forward(self, xs):
        return -xs

    def backward(self, gys):
        return -gys


def neg(x):
    return Neg()(x)


# 뺄셈
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gys):
        return gys, -gys


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


# 나눗셈
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gys):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gys / x1
        gx1 = gys * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


# 거듭 제곱
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, xs):
        y = xs ** self.c
        return y

    def backward(self, gys):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gys
        return gx


def pow(x, c):
    return Pow(c)(x)


Variable.__mul__ = mul
Variable.__add__ = add
Variable.__radd__ = add
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__div__ = div
Variable.__rdiv__ = rdiv
Variable.__pow__ = pow


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas1(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def matyas(x, y):
    z = sub(mul(0.26, add(pow(x, 2), pow(y, 2))), mul(0.48, mul(x, y)))


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
        (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z
