from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
import warnings

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    result_f_left = f(*[v - (epsilon if i == arg else 0) for i, v in enumerate(vals)])
    result_f_right = f(*[v + (epsilon if i == arg else 0) for i, v in enumerate(vals)])
    return (result_f_right - result_f_left) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    q = [variable]
    res = []
    visit_count = {}
    
    while q:
        u = q.pop()
        for v in u.parents:
            visit_count[v.unique_id] = visit_count.get(v.unique_id, 0) + 1
            if visit_count[v.unique_id] == 1:
                q.append(v)
    
    q.append(variable)
    while q:
        u = q.pop()
        if u.is_constant():
            continue
        res.append(u)
        for v in u.parents:
            visit_count[v.unique_id] -= 1
            if visit_count[v.unique_id] == 0:
                print(u, u.unique_id, v, v.unique_id)
                q.append(v)

    return res

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    order = topological_sort(variable)
    deriv_map = {}
    deriv_map.update({variable.unique_id: deriv})
    for u in order:
        if u.is_leaf():
            continue
        for v, d in u.chain_rule(deriv_map.get(u.unique_id)):
            print(u, u.unique_id, v, v.unique_id, d)
            if v.is_leaf():
                v.accumulate_derivative(d)
            else:
                if v.unique_id in deriv_map:
                    deriv_map[v.unique_id] += d
                else:
                    deriv_map.update({v.unique_id: d})


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
