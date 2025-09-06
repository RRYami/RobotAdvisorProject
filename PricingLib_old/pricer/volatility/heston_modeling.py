import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pricer.utilities.bin_calculator import bin_calculator

warnings.filterwarnings("ignore")
plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)


def get_data(ticker, numdays=7300):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=numdays)
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.sort_values(by="Date", ascending=False)
    return data


def log_return(data: pd.DataFrame):
    data["log_returns"] = np.log(data["Close"] / data["Close"].shift(1))
    data = data.dropna()
    return data


def create_plots(data: pd.DataFrame, path):
    textstr1 = "\n".join(
        (
            r"$\mu=%.4f$" % (data["log_returns"].mean(),),
            r"$\sigma=%.4f$" % (data["log_returns"].std(),),
        )
    )

    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5))
    ax[0, 0].plot(data["log_returns"])
    ax[0, 0].set_title("log_returns")
    ax[0, 1].hist(
        data["log_returns"],
        bins=bin_calculator(data["log_returns"]),
        density=True,
        histtype="step",
    )
    ax[0, 1].axvline(
        data["log_returns"].mean(),
        linestyle="dashed",
        linewidth=1,
        color="red",
        label="Mean",
    )
    ax[0, 1].text(
        0.05,
        0.81,
        textstr1,
        transform=ax[0, 1].transAxes,
        bbox=dict(facecolor="black", edgecolor="black"),
    )
    ax[0, 1].legend()
    ax[0, 1].set_ylabel("Density")
    ax[0, 1].set_title("Log Returns Histogram")
    plot_acf(data["log_returns"], ax=ax[1, 0])
    plot_pacf(data["log_returns"], ax=ax[1, 1])
    plt.savefig(path)
    plt.show()


def mu(x, dt, kappa, theta):
    ekt = np.exp(-kappa * dt)
    return x * ekt + theta * (1 - ekt)


def std(dt, kappa, sigma):
    e2kt = np.exp(-2 * kappa * dt)
    return sigma * np.sqrt((1 - e2kt) / (2 * kappa))


def log_likelihood_OU(theta_hat, x):
    kappa = theta_hat[0]
    theta = theta_hat[1]
    sigma = theta_hat[2]

    x_dt = x[1:]
    x_t = x[:-1]

    dt = 1 / 252

    mu_OU = mu(x_t, dt, kappa, theta)
    sigma_OU = std(dt, kappa, sigma)

    l_theta_hat = np.sum(np.log(sc.stats.norm.pdf(x_dt, loc=mu_OU, scale=sigma_OU)))

    return -l_theta_hat


def kappa_pos(theta_hat):
    kappa = theta_hat[0]
    return kappa


def sigma_pos(theta_hat):
    sigma = theta_hat[2]
    return sigma


def volatility_calc(data, rolling_window=42):
    data["volatility"] = (
        np.sqrt(252) * data["log_returns"].rolling(rolling_window).std()
    )
    volatility = data["volatility"].values
    volatility = volatility[~np.isnan(volatility)]
    return volatility


def calibration(volatility, guess_kappa, guess_theta, guess_sigma):
    cons_set = [{"type": "ineq", "fun": kappa_pos}, {"type": "ineq", "fun": sigma_pos}]
    theta0 = [guess_kappa, guess_sigma, guess_theta]
    opt = sc.optimize.minimize(
        fun=log_likelihood_OU,
        x0=theta0,
        method="Nelder-Mead",
        args=(volatility,),
        constraints=cons_set,
    )
    return opt.x
