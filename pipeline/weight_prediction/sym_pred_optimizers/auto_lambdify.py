from .sympy_optimizer import *
from sympy import lambdify
from collections import defaultdict


def lambdify_dict(coeff):
    """ Lambidfy the given expression, returns a dict describing result """
    free_symbols_list = sorted(coeff.free_symbols, key=str)
    f = lambdify(free_symbols_list, coeff, modules=['math'])

    free_symbols_names = sorted(map(str, free_symbols_list))
    generated_dict = dict(
        f=f, free_symbols=free_symbols_names, coeff_expr=coeff)
    return generated_dict


def auto_lambdify(max_staleness, optimizer_class, simplify=False):
    """ Auto generating functions for coefficients

    Example:
        Calculate the coefficnt of v.
        (Assuming "v" is in optimizer_class.collect_order)

        res, gap_res = auto_lambdify(...)

        # given stalnees some dict mapping parameters
        staleness = 1
        d = {"\\eta": 0.1, "\\gamma": 0.9}


        f = res[staleness]["v"]['f']
        required_args = res[staleness]["v"]["free_symbols"]
        # print(required_args)
        values = [d[a] for a in required_args]
        print(f(*values))

    """
    _, preds, gaps = run_sim(max_staleness, optimizer_class, simplify=simplify)
    res = defaultdict(dict)
    for idx, expr in enumerate(preds):
        curr_staleness = idx+1

        # print(f"Simplification ({curr_staleness})")
        expr = expr.expand()

        # Operations with the following symbols are exapnsive:
        symbols = optimizer_class.collect_order
        symbols = list(map(Symbol, symbols))

        for s in symbols:
            expr = expr.collect(s)

        for s in symbols:
            coeff = expr.coeff(s)
            if not coeff:
                raise NotImplementedError(
                    f"can't find {s} coeff in {expr}. Do it manually.")

            generated_dict = lambdify_dict(coeff)
            res[curr_staleness].update({str(s): generated_dict})

    # For gap 1
    gap_res = dict()
    grad = tplus_time("g", 0)
    gap1_expr = gaps[0].collect(grad).coeff(grad)

    generated_dict = lambdify_dict(gap1_expr)

    # free_symbols_list = sorted(gap1_expr.free_symbols)
    # f = lambdify(free_symbols_list , gap1_expr, modules=['math'])
    gap_res["gap_1"] = generated_dict

    return res, gap_res


if __name__ == "__main__":
    from pprint import pprint

    max_staleness = 3
    optimizer_class = WDSympySGDMsnag
    simplify = True
    res, gap_res = auto_lambdify(
        max_staleness, optimizer_class, simplify=simplify)

    pprint(res)
    pprint(gap_res)

    # Example:
    f = res[1]["v"]['f']
    required_args = res[1]["v"]["free_symbols"]
    print(required_args)
    d = {"\\eta": 0.1, "\\gamma": 0.9}
    values = [d[a] for a in required_args]
    print(f(*values))
