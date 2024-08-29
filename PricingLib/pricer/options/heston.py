import warnings

import numpy as np
from scipy.integrate import quad

warnings.filterwarnings("ignore")


# Heston characteristic function
def heston_charfunc(
    phi,
    initial_stock_price,
    initial_variance,
    kappa,
    theta,
    sigma,
    rho,
    lambd,
    tau,
    risk_free_rate,
):
    # constants
    a = kappa * theta
    b = kappa + lambd
    # common terms w.r.t phi
    rspi = rho * sigma * phi * 1j
    # define d parameter given phi and b
    d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 + (phi * 1j + phi**2) * sigma**2)
    # define g parameter given phi, b and d
    g = (b - rspi + d) / (b - rspi - d)
    # calculate characteristic function by components
    exp1 = np.exp(risk_free_rate * phi * 1j * tau)
    term2 = initial_stock_price ** (phi * 1j) * (
        (1 - g * np.exp(d * tau)) / (1 - g)
    ) ** (-2 * a / sigma**2)
    exp2 = np.exp(
        a * tau * (b - rspi + d) / sigma**2
        + initial_variance
        * (b - rspi + d)
        * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
        / sigma**2
    )
    return exp1 * term2 * exp2


# Heston integrand
def integrand(
    phi,
    initial_stock_price,
    strike,
    initial_variance,
    kappa,
    theta,
    sigma,
    rho,
    lambd,
    tau,
    risk_free_rate,
):
    args = (
        initial_stock_price,
        initial_variance,
        kappa,
        theta,
        sigma,
        rho,
        lambd,
        tau,
        risk_free_rate,
    )
    numerator = np.exp(risk_free_rate * tau) * heston_charfunc(
        phi - 1j, *args
    ) - strike * heston_charfunc(phi, *args)
    denominator = 1j * phi * strike ** (1j * phi)
    return numerator / denominator


# Heston price using rectangular integration
def heston_price_rec(
    initial_stock_price,
    strike,
    initial_variance,
    kappa,
    theta,
    sigma,
    rho,
    lambd,
    tau,
    risk_free_rate,
):
    args = (
        initial_stock_price,
        initial_variance,
        kappa,
        theta,
        sigma,
        rho,
        lambd,
        tau,
        risk_free_rate,
    )

    P, umax, N = 0, 100, 10000
    dphi = umax / N  # dphi is width

    for i in range(1, N):
        # rectangular integration
        phi = dphi * (2 * i + 1) / 2  # midpoint to calculate height
        numerator = np.exp(risk_free_rate * tau) * heston_charfunc(
            phi - 1j, *args
        ) - strike * heston_charfunc(phi, *args)
        denominator = 1j * phi * strike ** (1j * phi)
        P += dphi * numerator / denominator
    call = np.real(
        (initial_stock_price - strike * np.exp(-risk_free_rate * tau)) / 2 + P / np.pi
    )
    put = call - initial_stock_price + strike * np.exp(-risk_free_rate * tau)
    return [call, put]


# Heston price using quadrature integration
def heston_price(
    initial_stock_price,
    strike,
    initial_variance,
    kappa,
    theta,
    sigma,
    rho,
    lambd,
    tau,
    risk_free_rate,
):
    args = (
        initial_stock_price,
        strike,
        initial_variance,
        kappa,
        theta,
        sigma,
        rho,
        lambd,
        tau,
        risk_free_rate,
    )

    real_integral, err = np.real(quad(integrand, 0, 100, args=args))
    call = (
        initial_stock_price - strike * np.exp(-risk_free_rate * tau)
    ) / 2 + real_integral / np.pi
    put = call - initial_stock_price + strike * np.exp(-risk_free_rate * tau)
    return [call, put, err]
