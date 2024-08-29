from pricer.volatility.heston_modeling import (
    calibration,
    get_data,
    log_return,
    volatility_calc,
)

if __name__ == "__main__":
    ticker = "AAPL"
    data_a = get_data(ticker)
    data_b = log_return(data_a)
    vola = volatility_calc(data_b)
    res = calibration(vola, 1, 1, 1)
    print(res)
    print("done")
