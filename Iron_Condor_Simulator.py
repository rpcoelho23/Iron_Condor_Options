!pip install yfinance matplotlib numpy scipy pandas
# Cell 1: Install necessary library (if not already done)
# !pip install yfinance matplotlib numpy scipy pandas

# Cell 2: Import libraries
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import sys
import pandas as pd
from datetime import date, timedelta, datetime
import math # Added for sqrt in HV calculation

# Cell 3: Core Functions (Black-Scholes, Greeks, Payoff, Validation, Risk-Free Rate)

def get_risk_free_rate(ticker_symbol="^IRX"):
    """
    Fetches the latest yield for a given Treasury ticker from Yahoo Finance
    and returns it as a decimal. Defaults to 13-week T-Bill (^IRX).
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="5d")
        if history.empty:
            print(f"Warning: No recent history found for risk-free rate ticker {ticker_symbol}.")
            return None
        latest_yield_percent = history['Close'].iloc[-1]
        risk_free_rate_decimal = latest_yield_percent / 100.0
        # Keep this print statement as it was in the baseline
        # print(f"Fetched risk-free rate ({ticker_symbol}): {latest_yield_percent:.3f}% -> {risk_free_rate_decimal:.5f}")
        return risk_free_rate_decimal
    except Exception as e:
        print(f"Error fetching risk-free rate for ticker {ticker_symbol}: {e}")
        return None

def black_scholes_price(S, K, T, r, sigma, option_type):
    """Calculates Black-Scholes option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: # Added checks for S, K
        if option_type == "call": return max(0, S - K * np.exp(-r * (T if T>0 else 0)))
        elif option_type == "put": return max(0, K * np.exp(-r * (T if T>0 else 0)) - S)
        else: return 0
    try:
        with np.errstate(all='ignore'): # Suppress warnings during calculation
            d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call": price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type == "put": price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else: price = 0
        if not np.isfinite(price): raise ValueError("Calculation resulted in non-finite value")
        return price
    except (ValueError, OverflowError, ZeroDivisionError) as e:
         # Restore original print statement if desired
         # print(f"Warning: Math error in Black-Scholes for S={S}, K={K}, T={T}, r={r}, sigma={sigma}: {e}")
         if option_type == "call": return max(0, S - K)
         elif option_type == "put": return max(0, K - S)
         else: return 0

def option_greeks(S, K, T, r, sigma, option_type):
    """Calculates option Greeks (Delta, Gamma, Theta, Vega, Rho)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
         delta, gamma, theta, vega, rho = 0.0, 0.0, 0.0, 0.0, 0.0
         if T <= 0:
             if option_type == "call": delta = 1.0 if S > K else (0.5 if S == K else 0.0)
             elif option_type == "put": delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
         return delta, gamma, theta, vega, rho
    try:
        with np.errstate(all='ignore'): # Suppress warnings
            d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            pdf_d1 = norm.pdf(d1)

            gamma = pdf_d1 / (S * sigma * np.sqrt(T))
            vega = S * pdf_d1 * np.sqrt(T) * 0.01 # Per 1% change in vol
            theta_part1 = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))

            if option_type == "call":
                delta = norm.cdf(d1)
                theta = (theta_part1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0 # Per day
                rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) * 0.01 # Per 1% change in r
            elif option_type == "put":
                delta = norm.cdf(d1) - 1
                theta = (theta_part1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0 # Per day
                rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) * 0.01 # Per 1% change in r
            else: return 0, 0, 0, 0, 0

        if not all(np.isfinite([delta, gamma, theta, vega, rho])):
             # Restore original print statement if desired
             # print(f"Warning: Non-finite Greek calculated for S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
             return 0, 0, 0, 0, 0

        return delta, gamma, theta, vega, rho
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        # Restore original print statement if desired
        # print(f"Warning: Math error calculating Greeks for S={S}, K={K}, T={T}, r={r}, sigma={sigma}: {e}")
        return 0, 0, 0, 0, 0

def iron_condor_payoff(S_range, K1, K2, K3, K4, premium_short_call, premium_short_put, premium_long_call, premium_long_put):
    """Calculates the payoff of an Iron Condor strategy using actual premiums."""
    psc = premium_short_call if premium_short_call is not None else 0
    psp = premium_short_put if premium_short_put is not None else 0
    plc = premium_long_call if premium_long_call is not None else 0
    plp = premium_long_put if premium_long_put is not None else 0
    net_premium = psc + psp - plc - plp
    payoff = []
    for S in S_range:
        if K1 is None or K2 is None or K3 is None or K4 is None: payoff.append(np.nan); continue
        payoff_short_call = -max(0, S - K1)
        payoff_short_put = -max(0, K2 - S)
        payoff_long_call = max(0, S - K3)
        payoff_long_put = max(0, K4 - S)
        total_payoff = payoff_short_call + payoff_short_put + payoff_long_call + payoff_long_put + net_premium
        payoff.append(total_payoff)
    return np.array(payoff)

def validate_inputs(current_price, K1, K2, K3, K4, T, r, sigma, T_days_target, actual_exp_date_str):
    """Validates the input parameters for the Iron Condor, including fetched strikes."""
    errors = []
    today = date.today()
    actual_exp_date = datetime.strptime(actual_exp_date_str, '%Y-%m-%d').date()
    actual_days_to_exp = (actual_exp_date - today).days

    # --- >>> RESTORED PRINT STATEMENTS <<< ---
    print(f"Validation using: S={current_price:.2f}, K1={K1}, K2={K2}, K3={K3}, K4={K4}, T={T:.4f} ({actual_days_to_exp} days), r={r:.4f}, sigma={sigma:.4f}")
    print(f"Target Expiration: ~{T_days_target} days. Actual Expiration Used: {actual_exp_date_str} ({actual_days_to_exp} days)")
    # --- >>> END OF RESTORED PRINT STATEMENTS <<< ---

    if K1 is None or K2 is None or K3 is None or K4 is None: errors.append("FATAL: One or more strikes could not be determined.")
    elif not (isinstance(K1, (int, float)) and isinstance(K2, (int, float)) and isinstance(K3, (int, float)) and isinstance(K4, (int, float))): errors.append("FATAL: One or more strikes are not valid numbers.")
    elif not (K4 < K2 < K1 < K3): errors.append(f"FATAL: Selected strikes do not form a valid Iron Condor order! K4({K4}) < K2({K2}) < K1({K1}) < K3({K3}) is required.")

    if K1 is not None and K2 is not None and isinstance(K1, (int, float)) and isinstance(K2, (int, float)):
        if not (K2 < current_price): print(f"Info: Short Put strike K2 ({K2:.2f}) is not below the current price ({current_price:.2f}).")
        if not (K1 > current_price): print(f"Info: Short Call strike K1 ({K1:.2f}) is not above the current price ({current_price:.2f}).")

    if T <= 0: errors.append(f"FATAL: Calculated Time to Expiration (T={T:.4f}) is not positive.")
    if sigma is None or not isinstance(sigma, (int, float)): errors.append("FATAL: Volatility (sigma) is not a valid number.")
    elif sigma <= 0: errors.append("Volatility (sigma) must be positive.")
    elif sigma > 2.0: print(f"Warning: Input Volatility ({sigma*100:.1f}%) seems high (> 200%).")
    if r is None: errors.append("FATAL: Risk-free rate (r) could not be determined.")
    elif not isinstance(r, (int, float)): errors.append("FATAL: Risk-free rate (r) is not a valid number.")
    elif r < -0.1 or r > 0.2: print(f"Warning: Risk-free rate ({r*100:.1f}%) is outside the typical range (-10% to 20%).")

    return errors

def find_closest_expiration(available_dates, target_days):
    """Finds the expiration date string closest to the target number of days."""
    today = date.today()
    min_diff = float('inf')
    closest_date_str = None
    available_datetime_dates = []
    for date_str in available_dates:
        try:
            exp_date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            if exp_date_obj >= today: available_datetime_dates.append(exp_date_obj)
        except ValueError: continue
    if not available_datetime_dates: return None
    for exp_date in available_datetime_dates:
        diff = abs((exp_date - today).days - target_days)
        if diff < min_diff:
            min_diff = diff
            closest_date_str = exp_date.strftime('%Y-%m-%d')
        elif diff == min_diff and closest_date_str is not None:
             current_closest_date = datetime.strptime(closest_date_str, '%Y-%m-%d').date()
             if exp_date > current_closest_date: closest_date_str = exp_date.strftime('%Y-%m-%d')
    return closest_date_str

def get_option_details(options_df, current_price, position, option_type):
    """
    Finds the Nth OTM option and returns its strike, bid, ask, last, and IV.
    Note: Printing is handled in the main script now.
    """
    if options_df is None or options_df.empty: return None, None, None, None, None
    otm_options = pd.DataFrame()
    if option_type == 'call': otm_options = options_df[options_df['strike'] > current_price].sort_values(by='strike', ascending=True)
    elif option_type == 'put': otm_options = options_df[options_df['strike'] < current_price].sort_values(by='strike', ascending=False)
    else: return None, None, None, None, None
    if position <= 0: return None, None, None, None, None
    if len(otm_options) < position: return None, None, None, None, None
    selected_option = otm_options.iloc[position - 1]
    strike = selected_option['strike']
    bid = selected_option['bid']
    ask = selected_option['ask']
    last_price = selected_option['lastPrice']
    iv = selected_option['impliedVolatility']
    valid_bid = bid if pd.notna(bid) and bid > 0 else None
    valid_ask = ask if pd.notna(ask) and ask > 0 else None
    valid_last = last_price if pd.notna(last_price) and last_price > 0 else None
    valid_iv = iv if pd.notna(iv) and iv > 0 else None
    if valid_bid is None and valid_ask is None and valid_last is None: return strike, None, None, None, valid_iv
    return strike, valid_bid, valid_ask, valid_last, valid_iv

def calculate_historical_volatility(ticker_object, window=60):
    """Calculates annualized historical volatility over a given window."""
    try:
        history = ticker_object.history(period=f"{window + 20}d")
        if len(history) < window + 1: print(f"Warning: Insufficient history ({len(history)} days) for {window}-day HV."); return None
        history = history.iloc[-(window + 1):]
        log_returns = np.log(history['Close'] / history['Close'].shift(1))
        stdev = log_returns.iloc[1:].std()
        if stdev is None or not np.isfinite(stdev) or stdev <= 0: print(f"Warning: Invalid stdev ({stdev}) for HV."); return None
        annualized_hv = stdev * np.sqrt(252)
        return annualized_hv
    except Exception as e: print(f"Error calculating historical volatility: {e}"); return None

# Cell 4: User Inputs, Data Fetching, Option Selection, and Execution Logic

print("--- Iron Condor Strategy Simulator (Pulls Yfinance Market Data) ---")

# --- User Inputs ---
ticker = "AAPL"       # Stock Ticker
T_days_target = 30    # Target Time to Expiration (T) in Days
sigma_vol = 0.25      # Implied Volatility FOR GREEKS CALCULATION (as if there is an error fetching IVs)
CP1_pos = 2           # Position for Short Call (K1)
CP2_pos = 3           # Position for Long Call (K3) -> Must be > CP1_pos
PP1_pos = 1           # Position for Short Put (K2)
PP2_pos = 2           # Position for Long Put (K4) -> Must be > PP1_pos
hv_window = 60        # Days for HV calculation
# --- End User Inputs ---

# --- Data Fetching ---
print(f"\n--- Fetching Market Data for {ticker} ---")
current_price = 0
stock_data = None
historical_vol = None
try:
    stock_data = yf.Ticker(ticker)
    history = stock_data.history(period="5d")
    if not history.empty: current_price = history['Close'].iloc[-1]; print(f"Current Stock Price: ${current_price:.2f}")
    else: raise ValueError("History empty")
    historical_vol = calculate_historical_volatility(stock_data, window=hv_window)
except Exception as e: print(f"**FATAL Error fetching stock price or HV for ticker '{ticker}': {e}.**"); sys.exit("Cannot proceed.")

r_rate = get_risk_free_rate()
if r_rate is None: print("Warning: Failed risk-free rate fetch. Using default 0.05."); r_rate = 0.05
else: print(f"Risk-Free Rate (^IRX): {r_rate:.4f} ({r_rate*100:.2f}%)")

expirations = []; calls_df = None; puts_df = None; chosen_expiration_date_str = None; actual_days_to_exp = 0; T = 0
try:
    expirations = stock_data.options
    if not expirations: raise ValueError(f"No option expiration dates found.")
    chosen_expiration_date_str = find_closest_expiration(expirations, T_days_target)
    if chosen_expiration_date_str is None: raise ValueError(f"Could not find suitable future expiration near {T_days_target} days.")
    print(f"Target days: {T_days_target}. Closest available expiration: {chosen_expiration_date_str}")
    today = date.today(); exp_date = datetime.strptime(chosen_expiration_date_str, '%Y-%m-%d').date(); actual_days_to_exp = (exp_date - today).days
    if actual_days_to_exp <= 0: T = 0.0001; print(f"Warning: Days to expiration is {actual_days_to_exp}. Setting T small.")
    else: T = actual_days_to_exp / 365.0
    opt_chain = stock_data.option_chain(chosen_expiration_date_str)
    calls_df = opt_chain.calls; puts_df = opt_chain.puts
    if (calls_df is None or calls_df.empty) and (puts_df is None or puts_df.empty): raise ValueError(f"Option chain empty for {chosen_expiration_date_str}.")
except Exception as e: print(f"**FATAL Error fetching option data for {ticker}: {e}.**"); sys.exit("Cannot proceed.")

# --- Select Options and Premiums ---
print("\n--- Selecting Options Based on Position ---")
K1, K2, K3, K4 = None, None, None, None
premium_short_call_K1, premium_short_put_K2 = None, None
premium_long_call_K3, premium_long_put_K4 = None, None
iv_k1, iv_k2, iv_k3, iv_k4 = None, None, None, None
options_selected_successfully = True

# Short Call (K1)
strike_k1, bid_k1, ask_k1, last_k1, iv_k1 = get_option_details(calls_df, current_price, CP1_pos, 'call')
K1 = strike_k1
# --- >>> RESTORED PRINT STATEMENT <<< ---
print(f"Selected {CP1_pos}-th OTM call (Short K1): Strike={strike_k1 if strike_k1 else 'N/A'}, Bid={bid_k1}, Ask={ask_k1}, Last={last_k1}, IV={iv_k1 if iv_k1 else 'N/A'}")
if K1 is None: options_selected_successfully = False
else: premium_short_call_K1 = bid_k1 if bid_k1 is not None else last_k1
if premium_short_call_K1 is None and K1 is not None: print(f"FATAL: No premium for Short Call K1 (Strike {K1})."); options_selected_successfully = False

# Long Call (K3)
strike_k3, bid_k3, ask_k3, last_k3, iv_k3 = get_option_details(calls_df, current_price, CP2_pos, 'call')
K3 = strike_k3
# --- >>> RESTORED PRINT STATEMENT <<< ---
print(f"Selected {CP2_pos}-th OTM call (Long K3): Strike={strike_k3 if strike_k3 else 'N/A'}, Bid={bid_k3}, Ask={ask_k3}, Last={last_k3}, IV={iv_k3 if iv_k3 else 'N/A'}")
if K3 is None: options_selected_successfully = False
else: premium_long_call_K3 = ask_k3 if ask_k3 is not None else last_k3
if premium_long_call_K3 is None and K3 is not None: print(f"FATAL: No premium for Long Call K3 (Strike {K3})."); options_selected_successfully = False

# Short Put (K2)
strike_k2, bid_k2, ask_k2, last_k2, iv_k2 = get_option_details(puts_df, current_price, PP1_pos, 'put')
K2 = strike_k2
# --- >>> RESTORED PRINT STATEMENT <<< ---
print(f"Selected {PP1_pos}-th OTM put (Short K2): Strike={strike_k2 if strike_k2 else 'N/A'}, Bid={bid_k2}, Ask={ask_k2}, Last={last_k2}, IV={iv_k2 if iv_k2 else 'N/A'}")
if K2 is None: options_selected_successfully = False
else: premium_short_put_K2 = bid_k2 if bid_k2 is not None else last_k2
if premium_short_put_K2 is None and K2 is not None: print(f"FATAL: No premium for Short Put K2 (Strike {K2})."); options_selected_successfully = False

# Long Put (K4)
strike_k4, bid_k4, ask_k4, last_k4, iv_k4 = get_option_details(puts_df, current_price, PP2_pos, 'put')
K4 = strike_k4
# --- >>> RESTORED PRINT STATEMENT <<< ---
print(f"Selected {PP2_pos}-th OTM put (Long K4): Strike={strike_k4 if strike_k4 else 'N/A'}, Bid={bid_k4}, Ask={ask_k4}, Last={last_k4}, IV={iv_k4 if iv_k4 else 'N/A'}")
if K4 is None: options_selected_successfully = False
else: premium_long_put_K4 = ask_k4 if ask_k4 is not None else last_k4
if premium_long_put_K4 is None and K4 is not None: print(f"FATAL: No premium for Long Put K4 (Strike {K4})."); options_selected_successfully = False

# Calculate Average Implied Volatility
avg_iv = None
valid_ivs = [iv for iv in [iv_k1, iv_k2, iv_k3, iv_k4] if iv is not None and isinstance(iv, (int, float)) and iv > 0]
if len(valid_ivs) > 0: avg_iv = sum(valid_ivs) / len(valid_ivs)
else: print("Warning: Could not calculate Average IV from selected options.")

# Assign sigma for Greeks (using baseline logic: avg IV if possible, else input)
if avg_iv is not None: # Simplified check: if avg_iv was calculated, use it
    sigma = avg_iv
    # print(f"Using average fetched IV for Greeks calculation: {sigma:.4f}") # Keep original verbosity if desired
else:
    # print(f"Using user-provided sigma for Greeks: {sigma_vol:.4f}") # Keep original verbosity if desired
    sigma = sigma_vol

# --- Validation ---
print("\n--- Input Validation ---")
run_calculations = False
if not options_selected_successfully: print("FATAL: Failed to select one or more required options or their premiums. Cannot proceed.")
else:
    r = r_rate
    validation_errors = validate_inputs(current_price, K1, K2, K3, K4, T, r, sigma, T_days_target, chosen_expiration_date_str)
    premiums_valid = True
    if not all(p is not None and isinstance(p, (int, float)) for p in [premium_short_call_K1, premium_short_put_K2, premium_long_call_K3, premium_long_put_K4]):
         validation_errors.append("One or more premiums are missing or not valid numbers.")
         premiums_valid = False
    elif premiums_valid and (premium_short_call_K1 <= 0 or premium_short_put_K2 <= 0 or premium_long_call_K3 <= 0 or premium_long_put_K4 <= 0):
        validation_errors.append("One or more selected premiums are zero or negative.")

    if validation_errors:
        print("Input Validation Errors Found:")
        for error in validation_errors: print(f"- {error}")
        run_calculations = False
    else:
        # --- >>> RESTORED PRINT STATEMENT <<< ---
        print("Inputs and fetched options seem valid.")
        run_calculations = True

# --- Calculations and Plotting (Only if validation passes) ---
if run_calculations:
    print("\n--- Payoff Calculation (Using Market Premiums) ---")
    if K1 is None or K3 is None or K4 is None: print("FATAL: Cannot define plot range due to missing strikes."); run_calculations = False
    else:
        S_min = K4 * 0.90; S_max = K3 * 1.10; S_range = np.linspace(S_min, S_max, 500)
        payoff = iron_condor_payoff(S_range, K1, K2, K3, K4, premium_short_call_K1, premium_short_put_K2, premium_long_call_K3, premium_long_put_K4)
        net_premium_received = premium_short_call_K1 + premium_short_put_K2 - premium_long_call_K3 - premium_long_put_K4
        max_profit = net_premium_received
        max_loss = np.nan
        if (K2 - K4) > 0 and (K3 - K1) > 0:
             max_loss_put_side = -(K2 - K4) + net_premium_received; max_loss_call_side = -(K3 - K1) + net_premium_received
             max_loss = min(max_loss_put_side, max_loss_call_side)
        else: print("Warning: Cannot calculate Max Loss due to invalid strike wing widths.")
        breakeven_lower = K2 - net_premium_received; breakeven_upper = K1 + net_premium_received

        print(f"Selected Strikes: K1(ShortC)={K1:.2f}({CP1_pos}), K3(LongC)={K3:.2f}({CP2_pos}), K2(ShortP)={K2:.2f}({PP1_pos}), K4(LongP)={K4:.2f}({PP2_pos})")
        print(f"Premiums: ShortCall=${premium_short_call_K1:.2f}(Bid), ShortPut=${premium_short_put_K2:.2f}(Bid), LongCall=${premium_long_call_K3:.2f}(Ask), LongPut=${premium_long_put_K4:.2f}(Ask)")
        print(f"Net Premium Calculation: +${premium_short_call_K1:.2f} (Short Call) + ${premium_short_put_K2:.2f} (Short Put) - ${premium_long_call_K3:.2f} (Long Call) - ${premium_long_put_K4:.2f} (Long Put) = ${net_premium_received:.2f}")
        print(f"Total Credits: {round(premium_short_call_K1 + premium_short_put_K2,2)}")
        print(f"Total Debits: {round(premium_long_call_K3 + premium_long_put_K4,2)}")
        print(f"Maximum Profit: ${max_profit:.2f} (between K2=${K2:.2f} and K1=${K1:.2f})")
        if np.isfinite(max_loss): print(f"Maximum Loss: ${max_loss:.2f} (below K4=${K4:.2f} or above K3=${K3:.2f})")
        else: print(f"Maximum Loss: Calculation Error")
        print(f"Lower Breakeven Point: ${breakeven_lower:.2f}")
        print(f"Upper Breakeven Point: ${breakeven_upper:.2f}")

        print("\n--- Volatility Analysis ---")
        if historical_vol is not None and np.isfinite(historical_vol): print(f"{hv_window}-Day Historical Volatility (HV): {historical_vol:.4f} ({historical_vol*100:.2f}%)")
        else: print(f"{hv_window}-Day Historical Volatility (HV): Not Available")
        if avg_iv is not None and np.isfinite(avg_iv): 
            print(f"Average Implied Volatility (IV) of Selected Options: {avg_iv:.4f} ({avg_iv*100:.2f}%)")
        else: print(f"Average Implied Volatility (IV) of Selected Options: Not Available")
        weighted_vol = 0.7 * avg_iv + 0.3 * historical_vol
        print(f"Weighted Volatility (70% IV, 30% HV): {weighted_vol:.4f} ({weighted_vol*100:.2f}%)")
        iv_hv_ratio = None
        if avg_iv is not None and np.isfinite(avg_iv) and historical_vol is not None and np.isfinite(historical_vol) and historical_vol > 1e-6:
            iv_hv_ratio = avg_iv / historical_vol; 
            print(f"Implied/Historical Volatility Ratio (IV/HV): {iv_hv_ratio:.3f}")
            print("If IV/HV > 1: Implied Volatility (market's expectation of future volatility) is higher than recent Historical Volatility (actual past volatility).\n Options might be considered relatively 'expensive'. The market expects more movement than has recently occurred.")
        elif avg_iv is not None and historical_vol is not None and historical_vol <= 1e-6: print(f"Implied/Historical Volatility Ratio (IV/HV): Cannot calculate (HV is zero or negligible)")
        else: print(f"Implied/Historical Volatility Ratio (IV/HV): Not Available (Missing IV or HV)")

        print("\n--- Generating Payoff Diagram ---")
        fig_payoff, ax_payoff = plt.subplots(figsize=(10, 6))
        if not np.isnan(payoff).any():
            ax_payoff.plot(S_range, payoff, color='blue', linewidth=2, label="Iron Condor Payoff")
            ax_payoff.fill_between(S_range, payoff, 0, where=(payoff > 0), color='green', alpha=0.2, label='Profit Region')
            ax_payoff.fill_between(S_range, payoff, 0, where=(payoff < 0), color='red', alpha=0.2, label='Loss Region')
        else: ax_payoff.text(0.5, 0.5, 'Error in Payoff Calculation', ha='center', va='center', transform=ax_payoff.transAxes, color='red')
        ax_payoff.axhline(0, color="grey", linewidth=1, linestyle="--")
        ax_payoff.axvline(current_price, color="orange", linewidth=1, linestyle=":", label=f"Current Price (${current_price:.2f})")
        if K1 is not None and isinstance(K1, (int, float)): ax_payoff.axvline(K1, color="red", linewidth=1, linestyle="--", label=f"K1 (Short Call) ${K1:.2f}")
        if K2 is not None and isinstance(K2, (int, float)): ax_payoff.axvline(K2, color="red", linewidth=1, linestyle="--", label=f"K2 (Short Put) ${K2:.2f}")
        if K3 is not None and isinstance(K3, (int, float)): ax_payoff.axvline(K3, color="green", linewidth=1, linestyle="--", label=f"K3 (Long Call) ${K3:.2f}")
        if K4 is not None and isinstance(K4, (int, float)): ax_payoff.axvline(K4, color="green", linewidth=1, linestyle="--", label=f"K4 (Long Put) ${K4:.2f}")
        ax_payoff.set_xlabel("Stock Price at Expiration", fontsize=12); ax_payoff.set_ylabel("Profit / Loss ($)", fontsize=12)
        ax_payoff.set_title(f"Iron Condor Strategy Payoff for {ticker} (Exp: {chosen_expiration_date_str})", fontsize=14)
        if K1 is not None and K2 is not None and np.isfinite(max_profit):
             profit_zone_indices = np.where((S_range >= K2) & (S_range <= K1))[0]
             if len(profit_zone_indices) > 0:
                 mid_profit_zone_index = profit_zone_indices[len(profit_zone_indices)//2]
                 ax_payoff.annotate(f'Max Profit: ${max_profit:.2f}', xy=(S_range[mid_profit_zone_index], max_profit), xytext=(0, -20), textcoords='offset points', ha='center', va='top', arrowprops=dict(arrowstyle='->', color='green', shrinkA=5), bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6))
        if K3 is not None and K4 is not None and np.isfinite(max_loss) and not np.isnan(payoff).any():
             min_payoff_index = np.argmin(payoff)
             if min_payoff_index < len(S_range):
                  loss_text_x = S_range[min_payoff_index] if S_range[min_payoff_index] < K4+ (K3-K4)*0.1 or S_range[min_payoff_index] > K3 - (K3-K4)*0.1 else (S_min+K4)/2
                  ax_payoff.annotate(f'Max Loss: ${max_loss:.2f}', xy=(loss_text_x, max_loss), xytext=(0, -25 if loss_text_x < current_price else -25), textcoords='offset points', ha='center', va='top', arrowprops=dict(arrowstyle='->', color='red'), bbox=dict(boxstyle='round,pad=0.3', fc='pink', alpha=0.6))
        ax_payoff.legend(loc='best'); ax_payoff.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.show()

        print("\n--- Calculating and Plotting Greeks ---")
        greeks_k1 = [option_greeks(S, K1, T, r, sigma, "call") for S in S_range]; greeks_k2 = [option_greeks(S, K2, T, r, sigma, "put") for S in S_range]
        greeks_k3 = [option_greeks(S, K3, T, r, sigma, "call") for S in S_range]; greeks_k4 = [option_greeks(S, K4, T, r, sigma, "put") for S in S_range]
        delta_total = [-g[0] - g2[0] + g3[0] + g4[0] for g, g2, g3, g4 in zip(greeks_k1, greeks_k2, greeks_k3, greeks_k4)]
        gamma_total = [-g[1] - g2[1] + g3[1] + g4[1] for g, g2, g3, g4 in zip(greeks_k1, greeks_k2, greeks_k3, greeks_k4)]
        theta_total = [-g[2] - g2[2] + g3[2] + g4[2] for g, g2, g3, g4 in zip(greeks_k1, greeks_k2, greeks_k3, greeks_k4)]
        vega_total  = [-g[3] - g2[3] + g3[3] + g4[3] for g, g2, g3, g4 in zip(greeks_k1, greeks_k2, greeks_k3, greeks_k4)]
        rho_total   = [-g[4] - g2[4] + g3[4] + g4[4] for g, g2, g3, g4 in zip(greeks_k1, greeks_k2, greeks_k3, greeks_k4)]

        fig_greeks, ax_greeks = plt.subplots(3, 2, figsize=(12, 12)); fig_greeks.suptitle(f'Iron Condor Greeks for {ticker} (Exp: {chosen_expiration_date_str})', fontsize=16, y=1.02)
        if not np.isnan(delta_total).any() and not np.isinf(delta_total).any(): ax_greeks[0, 0].plot(S_range, delta_total, label="Total Delta", color='purple')
        ax_greeks[0, 0].set_title("Position Delta"); ax_greeks[0, 0].set_ylabel("Delta"); ax_greeks[0, 0].axhline(0, color='grey', linestyle='--', lw=0.8); ax_greeks[0, 0].axvline(current_price, color="orange", linestyle=":", lw=1, label=f"Current S ({current_price:.2f})"); ax_greeks[0, 0].grid(True, alpha=0.5); ax_greeks[0, 0].legend()
        if not np.isnan(gamma_total).any() and not np.isinf(gamma_total).any(): ax_greeks[0, 1].plot(S_range, gamma_total, label="Total Gamma", color='brown')
        ax_greeks[0, 1].set_title("Position Gamma"); ax_greeks[0, 1].set_ylabel("Gamma"); ax_greeks[0, 1].axhline(0, color='grey', linestyle='--', lw=0.8); ax_greeks[0, 1].axvline(current_price, color="orange", linestyle=":", lw=1); ax_greeks[0, 1].grid(True, alpha=0.5); ax_greeks[0, 1].legend()
        if not np.isnan(theta_total).any() and not np.isinf(theta_total).any(): ax_greeks[1, 0].plot(S_range, theta_total, label="Total Theta (per Day)", color='teal')
        ax_greeks[1, 0].set_title("Position Theta (per Day)"); ax_greeks[1, 0].set_ylabel("Theta"); ax_greeks[1, 0].axhline(0, color='grey', linestyle='--', lw=0.8); ax_greeks[1, 0].axvline(current_price, color="orange", linestyle=":", lw=1); ax_greeks[1, 0].grid(True, alpha=0.5); ax_greeks[1, 0].legend()
        if not np.isnan(vega_total).any() and not np.isinf(vega_total).any(): ax_greeks[1, 1].plot(S_range, vega_total, label="Total Vega (per 1% Vol)", color='magenta')
        ax_greeks[1, 1].set_title("Position Vega (per 1% Vol)"); ax_greeks[1, 1].set_ylabel("Vega"); ax_greeks[1, 1].axhline(0, color='grey', linestyle='--', lw=0.8); ax_greeks[1, 1].axvline(current_price, color="orange", linestyle=":", lw=1); ax_greeks[1, 1].grid(True, alpha=0.5); ax_greeks[1, 1].legend()
        if not np.isnan(rho_total).any() and not np.isinf(rho_total).any(): ax_greeks[2, 0].plot(S_range, rho_total, label="Total Rho (per 1% Rate)", color='lime')
        ax_greeks[2, 0].set_title("Position Rho (per 1% Rate)"); ax_greeks[2, 0].set_xlabel("Stock Price"); ax_greeks[2, 0].set_ylabel("Rho"); ax_greeks[2, 0].axhline(0, color='grey', linestyle='--', lw=0.8); ax_greeks[2, 0].axvline(current_price, color="orange", linestyle=":", lw=1); ax_greeks[2, 0].grid(True, alpha=0.5); ax_greeks[2, 0].legend()
        ax_greeks[2, 1].axis('off')
        sigma_source_text = f"σ={sigma*100:.1f}% (Avg IV)" if avg_iv is not None else f"σ={sigma*100:.1f}% (Input)" # Use avg_iv status to determine source
        textstr = '\n'.join((f'K1={K1:.2f}({CP1_pos}), K2={K2:.2f}({PP1_pos}), K3={K3:.2f}({CP2_pos}), K4={K4:.2f}({PP2_pos})', f'Exp: {chosen_expiration_date_str}', f'T={actual_days_to_exp} days ({T:.4f} yrs)', f'r={r*100:.2f}%, {sigma_source_text}' ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        if K1 is not None and K2 is not None and K3 is not None and K4 is not None: fig_greeks.text(0.75, 0.25, textstr, transform=fig_greeks.transFigure, fontsize=9, va='center', ha='center', bbox=props)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]); plt.show()
        print("\n--- Calculations Complete ---")

# This final else corresponds to the outer 'if run_calculations:'
else:
    if not run_calculations and options_selected_successfully: print("\n--- Calculations Skipped Due to Payoff Range or Plotting Error ---")
    elif not run_calculations: print("\n--- Calculations Skipped Due to Errors During Setup or Validation ---")
