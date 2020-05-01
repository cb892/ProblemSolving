   """""
    XlinFew = Xlin[:numDataPoints]
    YlinFew = Ylin[:numDataPoints]

    ExC = LSD.LinearLeastSquares(5,XlinFew,YlinFew)
    theta_few = ExC.theta()
    gamma_few = ExC.gamma()

    solver(theta_few,gamma_few,XlinFew,YlinFew)
    """

Matt:
Look up style guides for docstrings. I'd advise either the google or numpy style.

Note that as you have used PEP 484 type annotations, you don't have to repeat the type in the docstring, but I like to anyway as it is unobtrusive but is often the first place someone would look for a type definition. Until type definitions a-la PEP 484 become more common, I'd suggest doing both.

Here's an example for a simple function using the google style:

def greater_than(param1: int, param2: int) -> bool:
    """Checks whether the first parameter is greater than the second.

    Extended description of function would go here if needed.
    This is a pretty simple function so is probably unnecessary.

    Args:
        param1 (int): Description of what parameter 1 is.
        param2 (int): A well chosen variable name is an acceptable substitute for this.

    Returns:
        bool: True if param1 is greater than param2, False otherwise.

    """

    return param1 > param2
