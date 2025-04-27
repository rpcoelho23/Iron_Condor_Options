# **Iron Condor Options Simulator**
This Iron Condor Options Strategy Simulator is something I created to study the profitability of the Iron Condor Option Strategy.  It is regarded as one of the options strategy with the best return/risk ratios.  I created this code to study the profitability among different stocks and spread sizes using real market data pulled from yfinance.

To run just copy and paste the code in a Google Colab and execute it.  This is where the code was developed and tested.

## Concepts:

### How to Use for Comparison:
Run for AAPL: Set ticker = "AAPL" and run the entire script. Note down the printed values for:
XX-Day Historical Volatility (Annualized): (e.g., 0.2235 or 22.35%)
Average Implied Volatility of Selected Options: (e.g., 0.2450 or 24.50%)
Run for CVS: Change ticker = "CVS" and run the script again (keeping other inputs like T_days_target and OTM positions the same if you want a direct comparison for the same strategy setup). Note down its HV and Average IV values.
Compare: Look at the volatility numbers side-by-side.
The stock with the lower Average Implied Volatility is the one the market expects to be less volatile around your chosen strikes and expiration. This is often the more critical measure for an Iron Condor.
The stock with the lower Historical Volatility has been less volatile recently.
If both IV and HV are lower for one stock, it's a stronger candidate for a low-volatility strategy like the Iron Condor. If they diverge (e.g., low HV but high IV), investigate why the market might be expecting future volatility.

IV/HV > 1: Implied Volatility (market's expectation of future volatility) is higher than recent Historical Volatility (actual past volatility). Options might be considered relatively "expensive." The market expects more movement than has recently occurred.
IV/HV < 1: Implied Volatility is lower than recent Historical Volatility. Options might be considered relatively "cheap." The market expects less movement than has recently occurred.
IV/HV ≈ 1: Market expectation aligns reasonably well with recent history.
