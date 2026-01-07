import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

def get_options_chain(ticker_symbol):
    """Fetches the options chain for the nearest expiration."""
    tk = yf.Ticker(ticker_symbol)
    exps = tk.options
    
    # Get options for the next few expirations to ensure we have enough coverage
    # We focus on the nearest monthly or weekly for 'current' GEX, but aggregating a few is better.
    # For this simple script, let's take the first 2 expirations to capture near-term Gamma.
    options_data = []
    
    current_date = datetime.now().date()
    
    for date_str in exps[:4]: # Grab first 4 expirations
        try:
            opt = tk.option_chain(date_str)
            calls = opt.calls
            puts = opt.puts
            calls['type'] = 'call'
            puts['type'] = 'put'
            calls['expiration'] = date_str
            puts['expiration'] = date_str
            options_data.append(pd.concat([calls, puts]))
        except Exception as e:
            print(f"Error fetching expiration {date_str}: {e}")
            continue
            
    if not options_data:
        return pd.DataFrame() # Empty
        
    return pd.concat(options_data)

def black_scholes_gamma(S, K, T, r, sigma):
    """Calculates Gamma using Black-Scholes."""
    # S: Spot price
    # K: Strike price
    # T: Time to expiration (years)
    # r: Risk-free rate
    # sigma: Implied volatility
    
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def calculate_gex_profile(spot_price, options_df, interest_rate=0.045):
    """
    Calculates the Total Gamma Exposure (GEX) at various price levels to find the flip.
    Formula: Call GEX - Put GEX
    Call GEX = Gamma * OI * 100
    Put GEX = Gamma * OI * 100
    """
    # Filter for reasonable strikes to speed up (e.g., +/- 20% of spot)
    options_df = options_df[
        (options_df['strike'] > spot_price * 0.8) & 
        (options_df['strike'] < spot_price * 1.2)
    ].copy()
    
    # Pre-calculate time to expiration in years
    today = pd.Timestamp.now()
    options_df['T'] = (pd.to_datetime(options_df['expiration']) - today).dt.days / 365.0
    options_df['T'] = options_df['T'].clip(lower=0.001) # Avoid div by zero
    
    # Range of prices to test: spot +/- 10%
    price_range = np.linspace(spot_price * 0.9, spot_price * 1.1, 100)
    gex_values = []
    
    for price in price_range:
        total_gex = 0
        
        # Vectorized calculation would be faster, but loop is clearer for logic
        # Apply BS Gamma to each row
        # We use the row's Implied Volatility. If NaN, skip or assume something.
        
        df = options_df.dropna(subset=['impliedVolatility', 'openInterest']).copy()
        
        # Calculate d1 and Gamma for all rows at this 'price'
        sigma = df['impliedVolatility']
        K = df['strike']
        T = df['T']
        r = interest_rate
        
        d1 = (np.log(price / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gammas = norm.pdf(d1) / (price * sigma * np.sqrt(T))
        
        # GEX contribution
        # Call: + Gamma * OI * 100
        # Put: - Gamma * OI * 100
        
        call_mask = df['type'] == 'call'
        put_mask = df['type'] == 'put'
        
        gex = (gammas * df['openInterest'] * 100)
        total_gex = gex[call_mask].sum() - gex[put_mask].sum()
        
        gex_values.append(total_gex)
        
    return price_range, gex_values

def find_zero_crossing(x, y):
    """Finds the x value where y crosses zero."""
    # Convert to arrays
    x = np.array(x)
    y = np.array(y)
    
    # Find indices where sign changes
    signs = np.sign(y)
    sign_changes = ((np.roll(signs, 1) - signs) != 0).astype(int)
    sign_changes[0] = 0 # Ignore wrap-around
    
    indices = np.where(sign_changes == 1)[0]
    
    if len(indices) == 0:
        return None
        
    # Linearly interpolate the first crossing
    idx = indices[0]
    x1, x2 = x[idx-1], x[idx]
    y1, y2 = y[idx-1], y[idx]
    
    # y = mx + b
    m = (y2 - y1) / (x2 - x1)
    # 0 = m*x + b => x = -b/m = - (y1 - m*x1) / m = x1 - y1/m
    if m == 0: return x1
    zero_x = x1 - y1 / m
    return zero_x

def main():
    print("--- Starting Analysis ---")
    
    # 1. Fetch Data
    qqq = yf.Ticker("QQQ")
    mnq = yf.Ticker("MNQ=F")
    
    try:
        current_qqq = qqq.history(period="1d")['Close'].iloc[-1]
        print(f"Current QQQ Price: {current_qqq:.2f}")
    except IndexError:
        print("Could not fetch QQQ price. Market might be closed or data unavailable.")
        return

    try:
        current_mnq = mnq.history(period="1d")['Close'].iloc[-1]
        print(f"Current MNQ Price: {current_mnq:.2f}")
    except IndexError:
        print("Could not fetch MNQ price. Using approximation based on ratio if possible.")
        mnq_ratio = 1.0
        current_mnq = 0
    
    if current_mnq > 0:
        conversion_ratio = current_mnq / current_qqq
        print(f"QQQ to MNQ Ratio: {conversion_ratio:.4f}")
    else:
        print("Refetching MNQ...")
        # Sometimes continuous futures fail, try specific contract or just warn
        conversion_ratio = 0
    
    # 2. Daily/Weekly Expected Move
    # Use VIX-style calc: Spot * IV * sqrt(T)
    # But for precise Daily/Weekly, we should look at ATM IV for specific expirations.
    
    options = qqq.options
    # Weekly (closest to 7 days)
    # Daily (closest to 1 day)
    
    today = datetime.now().date()
    # Find closest expiration to today + 1 and today + 7
    
    # Helper to parse dates
    exp_dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in options]
    
    # Daily (1 day) - approximated by nearest expiration
    # If today is Friday, next is Monday (3 days).
    # We'll just take the nearest expiration for "Daily/Near Term" and scale if needed,
    # or just look for the specific IV.
    
    # Actually, simplistic Expected Move:
    # Daily EM = Price * (IV_Annual / sqrt(252))
    # Weekly EM = Price * (IV_Annual / sqrt(52))
    # We need a representative IV. Let's get ATM IV from the nearest expiration.
    
    near_term_exp = options[0] # Nearest
    opt_chain = qqq.option_chain(near_term_exp)
    calls = opt_chain.calls
    
    # Find ATM Call IV
    atm_call = calls.iloc[(calls['strike'] - current_qqq).abs().argsort()[:1]]
    iv_annual = atm_call['impliedVolatility'].values[0]
    
    daily_em = current_qqq * (iv_annual / np.sqrt(252))
    weekly_em = current_qqq * (iv_annual / np.sqrt(52))
    
    print(f"\nImplied Volatility (ATM, Exp {near_term_exp}): {iv_annual:.2%}")
    print(f"Daily Expected Move (Est): +/- ${daily_em:.2f}")
    print(f"Weekly Expected Move (Est): +/- ${weekly_em:.2f}")

    # 3. Gamma Flip
    print("\nCalculating Gamma Flip (this may take a moment)...")
    combined_chain = get_options_chain("QQQ")
    if not combined_chain.empty:
        xs, ys = calculate_gex_profile(current_qqq, combined_chain)
        flip_level = find_zero_crossing(xs, ys)
    else:
        flip_level = None
        
    if flip_level:
        print(f"Gamma Flip Level (QQQ): ${flip_level:.2f}")
    else:
        print("Could not calculate Gamma Flip (Data insufficient or no crossing found).")

    # 4. Translation & Display
    output_lines = []
    output_lines.append("\n" + "="*40)
    output_lines.append("      TRADING LEVELS REPORT      ")
    output_lines.append("="*40)
    
    levels = {
        "QQQ Spot": current_qqq,
        "QQQ Low Daily Exp": current_qqq - daily_em,
        "QQQ High Daily Exp": current_qqq + daily_em,
        "QQQ Low Weekly Exp": current_qqq - weekly_em,
        "QQQ High Weekly Exp": current_qqq + weekly_em,
        "Gamma Flip": flip_level if flip_level else 0
    }
    
    output_lines.append(f"{'Level':<20} | {'QQQ':<10} | {'MNQ':<10}")
    output_lines.append("-" * 46)
    
    for name, val in levels.items():
        if conversion_ratio > 0:
            mnq_val = val * conversion_ratio
            output_lines.append(f"{name:<20} | {val:<10.2f} | {mnq_val:<10.2f}")
        else:
            output_lines.append(f"{name:<20} | {val:<10.2f} | {'N/A':<10}")
            
    output_lines.append("="*40)
    
    if flip_level and conversion_ratio > 0:
        mnq_flip = flip_level * conversion_ratio
        output_lines.append("\n" + "*"*40)
        output_lines.append("  FOR TRADINGVIEW INDICATOR INPUT  ")
        output_lines.append("  Gamma Flip Level (MNQ): {:.2f}".format(mnq_flip))
        output_lines.append("*"*40 + "\n")
    else:
        mnq_flip = 0

    final_output = "\n".join(output_lines)
    print(final_output)
    
    with open("trading_levels_output.txt", "w") as f:
        f.write(final_output)

    # 5. Export to JSON for Web Dashboard
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "levels": {
            "Gamma Flip": mnq_flip,
            "Spot Price": current_mnq if current_mnq > 0 else (current_qqq * conversion_ratio if conversion_ratio else 0),
            "Daily High": (current_qqq + daily_em) * conversion_ratio if conversion_ratio else 0,
            "Daily Low": (current_qqq - daily_em) * conversion_ratio if conversion_ratio else 0,
            "Weekly High": (current_qqq + weekly_em) * conversion_ratio if conversion_ratio else 0,
            "Weekly Low": (current_qqq - weekly_em) * conversion_ratio if conversion_ratio else 0
        },
        "qqq_data": {
            "spot": current_qqq,
            "flip": flip_level if flip_level else 0,
            "daily_em": daily_em,
            "weekly_em": weekly_em
        }
    }
    
    import json
    with open("levels.json", "w") as f:
        json.dump(json_data, f, indent=4)
    print("Exported data to levels.json")

if __name__ == "__main__":
    main()
