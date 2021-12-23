import numpy as np
import weakref
import contextlib

import dezero


class Config:
    enable_backprop = True


class Variable:
    # 연산자 우선순위 - Variable 인스턴스의 연산자 우선순위를 ndarray 인스턴스의 연산자 우선순위 보다 높일 수 있음
    __array_priority__ = 200

    def __init__(self, data, name=None):
        # ndarray 만 취급하기
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은 (는) 지원하지 않습니다. ".format(type(data)))

        # data, grad 는 순전파 역전파 계산시에 사용, ndarray로 저장
        self.data = data
        self.grad = None  # 기울기

        self.name = name  # 변수 이름 지정
        self.creator = None
        self.generation = 0  # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 세대수를 기록한다.(부모 세대 + 1)

    # create_graph=False인 이유는 실무에서 역전파가 1회만 수행되는 경우가 압도적으로 맣기 때문에
    # 2차 이상의 미분이 필요시 True로 변경
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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
            # 역전파 계산 (메인 처리)
            gys = [output().grad for output in f.outputs]  # 1. 출력 변수인 outputs에 담겨있는 미분값들을 리스트에 담는다.

            if using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
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

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    @property
    def T(self):
        return dezero.functions.transpose(self)

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


class Function(object):
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        # 순전파 계산 (메인 처리)
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):  # 튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # 모드 전환
            self.generation = max([x.generation for x in inputs])
            # 연결을 만듦
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
    # 순전파 때 브로드캐스트가 일어나면 입력되는 x0, x1의 형상이 다를 것이다.
    # 두형상이 다를때 브로드캐스트용 역전파를 계산하는 것
    # 기울기 gx0는 x0의 형상이 되도록 합을 구하고 ,gx1도 똑같이 사용
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gys):
        gx0, gx1 = gys, gys
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)

        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y

    def backward(self, gys):
        x0, x1 = self.inputs
        gx0, gx1 = gys, gys
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx1 * x1, gx0 * x0


class Neg(Function):
    def forward(self, xs):
        return -xs

    def backward(self, gys):
        return -gys


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gys):
        gx0, gx1 = gys, gys
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)

        return gx0, -gx1


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y

    def backward(self, gys):
        x0, x1 = self.inputs
        gx0, gx1 = gys, gys
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        gx0 = gx0 / x1
        gx1 = gx1 * (-x0 / x1 ** 2)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, xs):
        y = xs ** self.c
        return y

    def backward(self, gys):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gys
        return gx


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


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__mul__ = mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow


class Parameter(Variable):
    pass
