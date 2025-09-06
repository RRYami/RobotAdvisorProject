from rich.console import Console
from rich.table import Table

from pricer.options.barone_adesi_whaley import BaroneAdesiWhaley
from pricer.options.black76 import Black76
from pricer.options.black_scholes import BlackScholes

console = Console()


def ui_menu():
    table = Table(title="Options Pricing Models", show_header=True, header_style="bold magenta")
    table.add_column("Model#", justify="center", style="dim", width=10)
    table.add_column("Model Name", justify="left", style="dim", width=30)
    table.add_column("Description", justify="left", style="dim", width=50)
    table.add_row("1.", "Black-Scholes Model", "General Black-Scholes Model for European Options")
    table.add_row("2.", "Black 76 Model", "General Black 76 Model for Futures Options")
    table.add_row("3.", "Barone Adesi Whaley Model", "Barone Adesi Whaley Model for American Options")
    console.print(table)
    console.print("\n")
    console.print("Select a pricing model by entering the corresponding [bold] Model# [/]:")


def ui_output(output: dict[str, str] | None):
    if output is None:
        console.print("Invalid input. Please try again.", style="bold red")
        return
    table = Table(title="Options Pricing Outputs", show_header=True, header_style="bold magenta")
    table.add_column("Output Name", justify="left", style="dim", width=20)
    table.add_column("Value", justify="right", style="dim", width=10)
    table.add_row("Call Price", "{:.6f}".format(output["call_price"]))
    table.add_row("Put Price", "{:.6f}".format(output["put_price"]))
    table.add_row("Call Delta", "{:.6f}".format(output["call_delta"]))
    table.add_row("Put Delta", "{:.6f}".format(output["put_delta"]))
    if "call_gamma" in output:
        table.add_row("Call Gamma", "{:.6f}".format(output["call_gamma"]))
        table.add_row("Put Gamma", "{:.6f}".format(output["put_gamma"]))
    else:
        table.add_row("Gamma", "{:.6f}".format(output["gamma"]))
    table.add_row("Call Theta", "{:.6f}".format(output["call_theta"]))
    table.add_row("Put Theta", "{:.6f}".format(output["put_theta"]))
    if "call_vega" in output:
        table.add_row("Call Vega", "{:.6f}".format(output["call_vega"]))
        table.add_row("Put Vega", "{:.6f}".format(output["put_vega"]))
    else:
        table.add_row("Vega", "{:.6f}".format(output["vega"]))
    table.add_row("Call Rho", "{:.6f}".format(output["call_rho"]))
    table.add_row("Put Rho", "{:.6f}".format(output["put_rho"]))
    console.print("\n")
    console.print(table)


def option_setup():
    console.print("Enter the option parameters:")
    console.print("\n")
    S = float(console.input("Spot price (S): "))
    K = float(console.input("Strike price (K): "))
    T = float(console.input("Time to maturity (T in days): "))
    r = float(console.input("Risk-free rate (r in decimal): "))
    sigma = float(console.input("Volatility (sigma in decimal): "))
    carry_cost = console.input("Cost of carry (if any in decimal): ")
    if carry_cost == "":
        carry_cost = None
    else:
        carry_cost = float(carry_cost)
    dividend = console.input("Dividend (if any in decimal): ")
    if dividend == "":
        dividend = 0.0
    else:
        dividend = float(dividend)
    return S, K, T, r, sigma, carry_cost, dividend


def option_future_setup():
    console.print("Enter the option parameters:")
    console.print("\n")
    future = float(console.input("Future price (F): "))
    K = float(console.input("Strike price (K): "))
    r = float(console.input("Risk-free rate (r in decimal): "))
    T_op = float(console.input("Time to maturity of the option (T in days): "))
    T_fut = float(console.input("Time to maturity of the future (T in days): "))
    sigma = float(console.input("Volatility (sigma in decimal): "))
    return future, K, r, T_op, T_fut, sigma


def pricing_model(model):
    if model == "1":
        S, K, T, r, sigma, _, dividend = option_setup()
        call_price = BlackScholes(S, K, r, T, sigma, dividend).call_price
        put_price = BlackScholes(S, K, r, T, sigma, dividend).put_price
        call_delta = BlackScholes(S, K, r, T, sigma, dividend).call_delta
        put_delta = BlackScholes(S, K, r, T, sigma, dividend).put_delta
        gamma = BlackScholes(S, K, r, T, sigma, dividend).gamma
        call_theta = BlackScholes(S, K, r, T, sigma, dividend).call_theta
        put_theta = BlackScholes(S, K, r, T, sigma, dividend).put_theta
        vega = BlackScholes(S, K, r, T, sigma, dividend).vega
        call_rho = BlackScholes(S, K, r, T, sigma, dividend).call_rho
        put_rho = BlackScholes(S, K, r, T, sigma, dividend).put_rho
        return {
            "call_price": call_price,
            "put_price": put_price,
            "call_delta": call_delta,
            "put_delta": put_delta,
            "gamma": gamma,
            "call_theta": call_theta,
            "put_theta": put_theta,
            "vega": vega,
            "call_rho": call_rho,
            "put_rho": put_rho,
        }
    elif model == "2":
        future, K, r, T_op, T_fut, sigma = option_future_setup()
        call_price = Black76(future, K, sigma, T_op, T_fut, r).call_price
        put_price = Black76(future, K, sigma, T_op, T_fut, r).put_price
        call_delta = Black76(future, K, sigma, T_op, T_fut, r).call_delta
        put_delta = Black76(future, K, sigma, T_op, T_fut, r).put_delta
        gamma = Black76(future, K, sigma, T_op, T_fut, r).gamma
        call_theta = Black76(future, K, sigma, T_op, T_fut, r).call_theta
        put_theta = Black76(future, K, sigma, T_op, T_fut, r).put_theta
        vega = Black76(future, K, sigma, T_op, T_fut, r).vega
        call_rho = Black76(future, K, sigma, T_op, T_fut, r).call_rho
        put_rho = Black76(future, K, sigma, T_op, T_fut, r).put_rho
        return {
            "call_price": call_price,
            "put_price": put_price,
            "call_delta": call_delta,
            "put_delta": put_delta,
            "gamma": gamma,
            "call_theta": call_theta,
            "put_theta": put_theta,
            "vega": vega,
            "call_rho": call_rho,
            "put_rho": put_rho,
        }
    elif model == "3":
        S, K, T, r, sigma, carry_cost, _ = option_setup()
        if carry_cost is None:
            carry_cost = 0.0
        call_price = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).american_price()[0]
        put_price = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).american_price()[1]
        call_delta = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).delta()[0]
        put_delta = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).delta()[1]
        call_gamma = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).gamma()[0]
        put_gamma = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).gamma()[1]
        call_theta = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).theta()[0]
        put_theta = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).theta()[1]
        call_vega = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).vega()[0]
        put_vega = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).vega()[1]
        call_rho = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).rho()[0]
        put_rho = BaroneAdesiWhaley(S, K, r, T, sigma, carry_cost).rho()[1]
        return {
            "call_price": call_price,
            "put_price": put_price,
            "call_delta": call_delta,
            "put_delta": put_delta,
            "call_gamma": call_gamma,
            "put_gamma": put_gamma,
            "call_theta": call_theta,
            "put_theta": put_theta,
            "call_vega": call_vega,
            "put_vega": put_vega,
            "call_rho": call_rho,
            "put_rho": put_rho,
        }


if __name__ == "__main__":
    while True:
        ui_menu()
        choice = console.input("Enter your choice: ")
        match choice:
            case "1":
                ui_output(pricing_model(choice))
            case "2":
                ui_output(pricing_model(choice))
            case "3":
                ui_output(pricing_model(choice))
            case _:
                console.print("Invalid choice. Please try again.", style="bold red")
        console.print("\n")
        user_input = console.input("[cyan]Do you want to continue?[/]([green]y[/]/[red]n[/]): ")
        if user_input.lower() in ["y", "yes"]:
            continue
        else:
            exit()
