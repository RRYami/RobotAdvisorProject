"""Module providingFunction to compute implied volatity of the underlying
of an option using newton's method."""

from pricer.options.black76 import Black76
from pricer.options.black_scholes import BlackScholes

BlackScholes(100, 100, 0.05, 1, 0.2).call_price  # Place holder for testing
Black76(100, 100, 0.2, 1, 1, 0.05).call_price  # Place holder for testing


def implied_vol_newton(class_name, spot, strike, rate, dte, callprice=None, putprice=None):
    """Definition:
    Newton-Raphson method for finding Implied Volatility (Root Method).
        Args:
            market_price (Float): Market Price of the Option
            OptType (String): Call or Put
            VolGuess (Float): Initial guess for volatility
            accuracy (Float) = 0.000001 : Accuracy of the Newton-Raphson method
            Iteration (Int) = 200000 : Number of iterations
        return:
            [implied_vol, i] (List): Implied Volatility and number of iterations
    """
    x0 = 1  # Initial guess
    h = 0.001  # Step size
    accuracy = 1e-8  # Accuracy of the Newton-Raphson method
    epsilon = 1e-14  # Required precision
    Iteration = 200000  # Number of iterations
    match callprice, putprice:
        case None, None:
            raise ValueError("Either callprice or putprice must be provided")
        case callprice, None:
            x1 = 0

            def f(x):
                return eval(class_name)(spot, strike, rate, dte, x).call_price - callprice

            for i in range(Iteration):
                y = f(x0)  # starting with initial guess
                yprime = (f(x0 + h) - f(x0 - h)) / (2 * h)  # central difference, the derivative of the function
                if abs(yprime) < epsilon:  # stop if the denominator is too small
                    break
                x1 = x0 - y / yprime  # perform Newton's computation
                if abs(x1 - x0) <= accuracy * abs(x1):  # stop when the result is within the desired tolerance
                    break
                x0 = x1  # update x0 to start the process again
            return x1
        case None, putprice:
            x1 = 0

            def g(x):
                return eval(class_name)(spot, strike, rate, dte, x).put_price - putprice

            for i in range(Iteration):
                y = g(x0)  # starting with initial guess
                yprime = (g(x0 + h) - g(x0 - h)) / (2 * h)  # central difference, the derivative of the function
                if abs(yprime) < epsilon:  # stop if the denominator is too small
                    break
                x1 = x0 - y / yprime  # perform Newton's computation
                if abs(x1 - x0) <= accuracy * abs(x1):  # stop when the result is within the desired tolerance
                    break
                x0 = x1  # update x0 to start the process again
            return x1


def implied_vol_bisection(
    class_name,
    spot,
    strike,
    rate,
    dte,
    callprice=None,
    putprice=None,
    high=1000.0,
    low=0.0,
):
    price = 0  # Initialisation of price
    if callprice:
        price = callprice
    if putprice and not callprice:
        price = putprice
    accuracy = 1e-8  # Accuracy
    iteration = 200000  # Number of iterations
    estimate = 0  # Initialisation of estimate
    mid = 0  # Initialisation of mid
    for i in range(iteration):
        mid = (high + low) / 2.0
        if mid < accuracy:
            mid = accuracy
        if callprice:
            estimate = eval(class_name)(spot, strike, rate, dte, mid).call_price
        if putprice and not callprice:
            estimate = eval(class_name)(spot, strike, rate, dte, mid).put_price
        if round(estimate, 8) == round(price, 8):
            break
        elif estimate > price:
            high = mid
        elif estimate < price:
            low = mid
    return mid


def implied_vol_black_newton(
    class_name,
    future_price,
    strike,
    bit_t,
    prompt_t,
    rate,
    callprice=None,
    putprice=None,
):
    """Definition:
    Newton-Raphson method for finding Implied Volatility (Root Method).
        Args:
            market_price (Float): Market Price of the Option
            OptType (String): Call or Put
            VolGuess (Float): Initial guess for volatility
            accuracy (Float) = 0.000001 : Accuracy of the Newton-Raphson method
            Iteration (Int) = 200000 : Number of iterations
        return:
            [implied_vol, i] (List): Implied Volatility and number of iterations
    """
    x0 = 1  # Initial guess
    h = 0.001  # Step size
    accuracy = 1e-8  # Accuracy of the Newton-Raphson method
    epsilon = 1e-14  # Required precision
    Iteration = 200000  # Number of iterations
    match callprice, putprice:
        case None, None:
            raise ValueError("Either callprice or putprice must be provided")
        case callprice, None:
            x1 = 0

            def f(x):
                return eval(class_name)(future_price, strike, x, bit_t, prompt_t, rate).call_price - callprice

            for i in range(Iteration):
                y = f(x0)  # starting with initial guess
                yprime = (f(x0 + h) - f(x0 - h)) / (2 * h)  # central difference, the derivative of the function
                if abs(yprime) < epsilon:  # stop if the denominator is too small
                    break
                x1 = x0 - y / yprime  # perform Newton's computation
                if abs(x1 - x0) <= accuracy * abs(x1):  # stop when the result is within the desired tolerance
                    break
                x0 = x1  # update x0 to start the process again
            return x1
        case None, putprice:
            x1 = 0

            def g(x):
                return eval(class_name)(future_price, strike, x, bit_t, prompt_t, rate).put_price - putprice

            for i in range(Iteration):
                y = g(x0)  # starting with initial guess
                yprime = (g(x0 + h) - g(x0 - h)) / (2 * h)  # central difference, the derivative of the function
                if abs(yprime) < epsilon:  # stop if the denominator is too small
                    break
                x1 = x0 - y / yprime  # perform Newton's computation
                if abs(x1 - x0) <= accuracy * abs(x1):  # stop when the result is within the desired tolerance
                    break
                x0 = x1  # update x0 to start the process again
            return x1


def implied_vol_black_bisection(
    class_name,
    future_price,
    strike,
    bit_t,
    prompt_t,
    rate,
    callprice=None,
    putprice=None,
    high=1000.0,
    low=0.0,
):
    price = 0  # Initialisation of price
    if callprice:
        price = callprice
    if putprice and not callprice:
        price = putprice
    accuracy = 1e-8  # Accuracy
    iteration = 200000  # Number of iterations
    estimate = 0  # Initialisation of estimate
    mid = 0  # Initialisation of mid
    for i in range(iteration):
        mid = (high + low) / 2.0
        if mid < accuracy:
            mid = accuracy
        if callprice:
            estimate = eval(class_name)(future_price, strike, mid, bit_t, prompt_t, rate).call_price
        if putprice and not callprice:
            estimate = eval(class_name)(future_price, strike, mid, bit_t, prompt_t, rate).put_price
        if round(estimate, 8) == round(price, 8):
            break
        elif estimate > price:
            high = mid
        elif estimate < price:
            low = mid
    return mid
