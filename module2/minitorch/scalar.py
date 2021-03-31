from .autodiff import FunctionBase, Variable, History
from . import operators
import numpy as np
import math

## Task 1.1
## Derivatives


def central_difference(f, *vals, arg=0, epsilon=1e-6):
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
       f : arbitrary function from n-scalar args to one value
       *vals (floats): n-float values :math:`x_0 \ldots x_{n-1}`
       arg (int): the number :math:`i` of the arg to compute the derivative
       epsilon (float): a small constant

    Returns:
       float : An approximation of :math:`f'_i(x_0, \ldots, x_{n-1})`
    """
    # TODO: Implement for Task 1.1.
    x = list(vals)
    y = list(vals)
    x[arg] += epsilon
    y[arg] -= epsilon
    difference = f(*x) - f(*y)
    output = difference / (2 * epsilon)
    return output


## Task 1.2 and 1.4
## Scalar Forward and Backward


class Scalar(Variable):
    """
    A reimplementation of scalar values for autodifferentiation
    tracking.  Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    :class:`ScalarFunction`.

    Attributes:
        data (float): The wrapped scalar value.

    """

    def __init__(self, v, back=History(), name=None):
        super().__init__(back, name=name)
        self.data = float(v)

    def __repr__(self):
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        # print(f'__mul__ operator self type: {type(self)}')
        return Mul.apply(self, b)

    def __truediv__(self, b):
        # print(f'Within True Div self type: {type(self)}')
        y = Inv.apply(b)
        # if y.history is None:
        #    print(f'NO History: a = {self.data} b = {b}')
        # print(f'Inverse result type: {type(y)}')
        # print(f'Resulting Inverse history type : {(type(y.history))}')
        # print(f'Resulting Inverse history last function: {(y.history.last_fn)}')
        # print(f'Resulting Inverse history: {(y.history.inputs)}')
        x = Mul.apply(self, y)
        # print(f'Resulting multiply history: {(x.history.last_fn)}')
        return x

    def __add__(self, b):
        # TODO: Implement for Task 1.2.
        return Add.apply(self, b)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def __lt__(self, b):
        return LT.apply(self, b)
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    def __gt__(self, b):
        return not LT.apply(self, b)
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    def __sub__(self, b):
        return Add.apply(self, -1.0 * b)
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    def __neg__(self):
        return Neg.apply(self)
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    def log(self):
        # TODO: Implement for Task 1.2.
        return Log.apply(self)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def exp(self):
        return Exp.apply(self)
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    def sigmoid(self):
        # TODO: Implement for Task 1.2.
        return Sigmoid.apply(self)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def relu(self):
        # print(f'Scalar Relu Value: {self}')
        return ReLU.apply(self)
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    def get_data(self):
        return self.data


class ScalarFunction(FunctionBase):
    "A function that processes and produces Scalar variables."

    @staticmethod
    def forward(ctx, *inputs):
        """Args:

           ctx (:class:`Context`): A special container object to save
                                   any information that may be needed for the call to backward.
           *inputs (list of numbers): Numerical arguments.

        Returns:
            number : The computation of the function :math:`f`

        """
        pass

    @staticmethod
    def backward(ctx, d_out):
        """
        Args:
            ctx (Context): A special container object holding any information saved during in the corresponding `forward` call.
            d_out (number):
        Returns:
            numbers : The computation of the derivative function :math:`f'_{x_i}` for each input :math:`x_i` times `d_out`.
        """
        pass

    # checks.
    variable = Scalar
    data_type = float

    @staticmethod
    def data(a):
        return a


# Examples
class Add(ScalarFunction):
    "Addition function"

    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, d_output):
        # print('ADD BACKWARD FUNCTION')

        return d_output, d_output


class Log(ScalarFunction):
    "Log function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx, d_output):
        # print('LOG BACKWARD FUNCTION')

        a = ctx.saved_values
        return operators.log_back(a, d_output)


class LT(ScalarFunction):
    "Less-than function"

    @staticmethod
    def forward(ctx, a, b):
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx, d_output):
        # print('LT BACKWARD FUNCTION')

        return 0.0


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        # print(f'Multiply variable Type: {type(a)} {type(b)}')
        return a * b
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # print('MULTIPLY BACKWARD FUNCTION')
        x, y = ctx.saved_values
        return y * d_output, x * d_output
        # TODO: Implement for Task 1.4.
        # raise NotImplementedError('Need to implement for Task 1.4')


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)

        # print(f'Inverse Value Type: {type(a)}')
        # a.history = History(Inv, ctx, [a])

        return operators.inv(a)
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # print('INVERSE BACKWARD FUNCTION')
        x = ctx.saved_values
        return operators.inv_back(x, d_output)
        # TODO: Implement for Task 1.4.
        # raise NotImplementedError('Need to implement for Task 1.4')


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx, a):
        # ctx.save_for_backward(a)
        return operators.neg(a)
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # print('NEGATIVE BACKWARD FUNCTION')
        return -1.0 * d_output
        # TODO: Implement for Task 1.4.
        # raise NotImplementedError('Need to implement for Task 1.4')


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(operators.sigmoid(a))
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx, d_output):
        x = ctx.saved_values
        return d_output * x * (1 - x)


class ReLU(ScalarFunction):
    "ReLU function"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx, d_output):
        x = ctx.saved_values
        return operators.relu_back(x, d_output)


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return math.exp(a)
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # print('EXP BACKWARD FUNCTION')
        x = ctx.saved_values
        return d_output * math.exp(x)
        # TODO: Implement for Task 1.4.
        # raise NotImplementedError('Need to implement for Task 1.4')


def derivative_check(f, *scalars):
    # print(f'Number of Scalars: {len(scalars)}')
    for x in scalars:
        x.requires_grad_(True)
    out = f(*scalars)
    # print(f'Out: {out}')
    # print(f'Number of Inputs: {len(out.history.inputs)}')
    # for x in out.history.inputs:
    #    if not x.history.is_leaf():

    #    print(f'Inputs to out: {x}, Inputs inputs : {x.history.inputs} Inputs last function : {x.history.last_fn}')

    out.backward()
    # return
    vals = [v for v in scalars]
    # print(f'Scalars: {scalars}')
    for i, x in enumerate(scalars):
        check = central_difference(f, *vals, arg=i)
        # print(f'x = {x} is leaf? : {x.history.is_leaf()} Derivative = {x.derivative}, check = {check}')
        np.testing.assert_allclose(x.derivative, check.data, 1e-2, 1e-2)
