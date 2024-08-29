import numpy as np
import sympy as sp
import pandas as pd
import sqlalchemy as db

engine = db.create_engine(
    "mssql+pyodbc:///?odbc_connect=Driver={SQL Server};Server=192.168.11.102;Database=IDPM;Trusted_Connection=True;"
)

discount_factor = pd.read_excel(
    r"C:\Users\rrenard\Desktop\CMS_data\5YearDiscount_OIS.xlsx"
)
discount_factor_libor = pd.read_excel(
    r"C:\Users\rrenard\Desktop\CMS_data\10YearDiscount_LIBOR_usd.xlsx", index_col=0
)
vols = pd.read_excel(r"C:\Users\rrenard\Desktop\CMS_data\VolsForCalc.xlsx", index_col=0)

discount_factor_array = np.array(discount_factor["Discount"])
discount_factor_array = np.delete(discount_factor_array, 0)
discount_factor_libor_array = np.array(discount_factor_libor["Discount"])
discount_factor_libor_array = np.delete(discount_factor_libor_array, 0)


def par_swap_rate_calculation():
    par_swap_rate_list = []
    for i in range(0, 11):
        par_swap_rate = (1 - discount_factor_libor_array[i]) / sum(
            discount_factor_libor_array[: i + 1]
        )
        par_swap_rate_list.append(par_swap_rate)
    return par_swap_rate_list


def number_extraction(list_object):
    first = [str(element) for element in list_object]
    intermediate = [element.split("*") for element in first]
    result = [element[0] for element in intermediate]
    return result


# expressing a Bond as a function of x (x = yield to maturity)
def bond_part(coupon, coupon_frequency, time_to_maturity):
    gxmid = 0
    x = sp.Symbol("x")

    if coupon_frequency == 1 / 2:
        TimeToMaturityMultiplier = 2
    elif coupon_frequency == 1 / 4:
        TimeToMaturityMultiplier = 4
    else:
        TimeToMaturityMultiplier = 1

    n = (TimeToMaturityMultiplier * time_to_maturity) + 1

    for i in np.arange(1.0, n, 1.0):
        gx1 = coupon / (1 + coupon_frequency * x) ** i
        gxmid += gx1
    gx2 = 100 / (1 + coupon_frequency * x) ** (
        time_to_maturity * TimeToMaturityMultiplier
    )
    gxFinal = gxmid + gx2
    return gxFinal


# Computing the first and second derivative of a Bond
def derivatives_bond_part(coupon, coupon_frequency, time_to_maturity, evaluate_by):
    gxFinal = bond_part(coupon, coupon_frequency, time_to_maturity)
    x = sp.Symbol("x")
    gx_prime_result_list = []
    gx_double_prime_result_list = []
    for i in range(len(evaluate_by)):
        gx_prime = gxFinal.diff(x)
        gx_double_prime = gx_prime.diff(x)
        gx_prime_result = gx_prime.subs(x, evaluate_by[i])
        gx_prime_result_list.append(gx_prime_result)
        gx_double_prime_result = gx_double_prime.subs(x, evaluate_by[i])
        gx_double_prime_result_list.append(gx_double_prime_result)
    return gx_prime_result_list, gx_double_prime_result_list


# convexity adjustment with timing adjustment
def convexity_timing_adj(
    coupon,
    coupon_frequency,
    time_to_maturity,
    evaluate_by,
    coupon_rate,
    swaption_impl_vol,
    forward_interest_rate,
    tau,
    corr_forw_swap_rate_forw_int_rate,
    caplet_impl_vol,
):
    ti = sp.Symbol("ti")
    convex_time_adj_list = []
    for i, j in zip(range(len(swaption_impl_vol)), range(len(caplet_impl_vol))):
        part_1 = (
            0.5
            * (coupon_rate**2)
            * (swaption_impl_vol[i] ** 2)
            * ti
            * (
                derivatives_bond_part(
                    coupon, coupon_frequency, time_to_maturity, evaluate_by
                )[1][0]
                / derivatives_bond_part(
                    coupon, coupon_frequency, time_to_maturity, evaluate_by
                )[0][0]
            )
        )

        part_2 = (
            coupon_rate
            * tau
            * forward_interest_rate
            * corr_forw_swap_rate_forw_int_rate
            * swaption_impl_vol[i]
            * caplet_impl_vol[j]
            * ti
        ) / (1 + forward_interest_rate * tau)

        convex_time_adj = coupon_rate - part_1 - part_2
        convex_time_adj_list.append(convex_time_adj)
    return convex_time_adj_list


# convexity adjustment
def convexity_adj(
    coupon, coupon_frequency, time_to_maturity, evaluate_by, swaption_impl_vol
):
    ti = sp.Symbol("ti")
    convex_time_adj_list = []
    par_swap_rate = par_swap_rate_calculation()
    for i, j, k in zip(
        range(len(swaption_impl_vol)),
        range(len(par_swap_rate)),
        range(len(evaluate_by)),
    ):
        convex_time_adj = par_swap_rate[j] - (
            0.5
            * (par_swap_rate[j] ** 2)
            * (swaption_impl_vol[i] ** 2)
            * ti
            * (
                derivatives_bond_part(
                    coupon, coupon_frequency, time_to_maturity, evaluate_by
                )[1][k]
                / derivatives_bond_part(
                    coupon, coupon_frequency, time_to_maturity, evaluate_by
                )[0][k]
            )
        )
        convex_time_adj_list.append(convex_time_adj)
    return convex_time_adj_list


# Swap Value using convexity adjustment and timing adjustment
def swap_value(
    notional,
    coupon,
    coupon_frequency,
    time_to_maturity,
    evaluate_by,
    coupon_rate,
    swaption_impl_vol,
    forward_interest_rate,
    tau,
    corr_forw_swap_rate_forw_int_rate,
    caplet_impl_vol,
):
    n = time_to_maturity + coupon_frequency
    m = time_to_maturity * 2
    total = 0
    result = 0.0

    ConvexAdj = convexity_timing_adj(
        coupon,
        coupon_frequency,
        time_to_maturity,
        evaluate_by,
        coupon_rate,
        swaption_impl_vol,
        forward_interest_rate,
        tau,
        corr_forw_swap_rate_forw_int_rate,
        caplet_impl_vol,
    )

    res = number_extraction(ConvexAdj)

    for i, j, k in zip(
        np.arange(0.0, n, coupon_frequency),
        np.arange(0, m + 1, 1),
        np.arange(0, len(res) + 1, 1),
    ):
        result = (float(res[k]) * i * notional) * discount_factor_libor["Discount"][j]
        # IDK about this fixe rate, it is not clear in Hull
    total += result
    return total


# Swap Value using convexity adjustment
def swap_value_no_timing(
    notional, coupon, coupon_frequency, time_to_maturity, evaluate_by, swaption_impl_vol
):
    n = time_to_maturity + coupon_frequency
    m = time_to_maturity * 2
    total = 0

    ConvexAdj = convexity_adj(
        coupon, coupon_frequency, time_to_maturity, evaluate_by, swaption_impl_vol
    )
    res = number_extraction(ConvexAdj)

    for i, j, k in zip(
        np.arange(0.0, n, coupon_frequency),
        np.arange(0, m + 1, 1),
        np.arange(0, len(res), 1),
    ):
        result = (float(res[k]) * i * notional) * discount_factor["Discount"][j]
        total += result
    return total


par_swaps = par_swap_rate_calculation()
s2 = swap_value_no_timing(10000000, 2.5, 0.5, 5, par_swaps, list(vols["Swaption"]))
print(s2)
