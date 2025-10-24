# strategies/stg_TEST_1.py
import numpy as np
import pandas as pd
from backtesting import Strategy

def _roll_mean(arr, n):
    return pd.Series(arr).rolling(n, min_periods=n).mean().values

def _roll_std(arr, n):
    return pd.Series(arr).rolling(n, min_periods=n).std(ddof=0).values

class TEST_1(Strategy):
    """
    Z-score vs Rolling Mean (ALL-IN / ALL-OUT)
    
    Logic:
      - z = (price - rolling_mean) / rolling_std
      - Enter long (ALL IN) when:
          z <= -z_entry
          AND rolling_std <= vol_limit  ← new condition
      - Exit long (ALL OUT) when z >= -z_exit

    Optional shorts if allow_short=True.
    """

    # --- Tunable parameters ---
    window: int    = 10
    z_entry: float = 2.0
    z_exit:  float = 0.5
    vol_limit: float = 0.02   # ⬅️ new volatility filter
    allow_short: bool = False
    min_std: float = 1e-8

    # sizing mode
    size_mode: str = "units"    # 'units' | 'fraction'
    fraction_all_in: float = 0.999

    def init(self):
        price = self.data.Close
        self.rm = self.I(_roll_mean, price, self.window)
        self.rs = self.I(_roll_std,  price, self.window)

        def _zscore(x, m, s, eps):
            s_safe = np.maximum(s, eps)
            return (x - m) / s_safe

        self.z = self.I(_zscore, price, self.rm, self.rs, self.min_std)

    # ---- ALL-IN helpers ----
    def _all_in_long(self, price_now: float):
        if self.size_mode == "fraction":
            self.buy(size=self.fraction_all_in)
        else:
            units = int(self.equity // price_now)
            if units >= 1:
                self.buy(size=units)

    def _all_in_short(self, price_now: float):
        if self.size_mode == "fraction":
            self.sell(size=self.fraction_all_in, short=True)
        else:
            units = int(self.equity // price_now)
            if units >= 1:
                self.sell(size=units, short=True)

    def next(self):
        z_now = float(self.z[-1])
        vol_now = float(self.rs[-1])   # current rolling std
        if np.isnan(z_now) or np.isnan(vol_now):
            return

        price_now = float(self.data.Close[-1])

        # --- Manage open positions (ALL OUT) ---
        if self.position:
            if self.position.is_long and z_now >= -self.z_exit:
                self.position.close()
                return
            if self.allow_short and self.position.is_short and z_now <= self.z_exit:
                self.position.close()
                return
            return  # holding

        # --- Flat: Entries (ALL IN) ---
        # ✅ Added volatility filter here
        if z_now <= -self.z_entry and vol_now <= self.vol_limit:
            self._all_in_long(price_now)
            return

        if self.allow_short and z_now >= self.z_entry and vol_now <= self.vol_limit:
            self._all_in_short(price_now)
