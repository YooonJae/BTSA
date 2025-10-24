from backtesting import Strategy
import pandas as pd
import numpy as np

class ZRM_1(Strategy):
    """
    Long-only regime strategy using Z-score vs rolling mean.
    - Enter LONG when price is far BELOW its rolling mean (z <= -z_entry)
    - Exit LONG when deviation mean-reverts toward zero (z > -z_exit)
    """
    window = 100       # rolling window length
    z_entry = 2.5      # threshold for long entry (below mean)
    z_exit = 0.3       # threshold for exit (near zero)

    def init(self):
        price = self.data.Close

        # Rolling mean & std via self.I (vectorized functions expected)
        self.mean = self.I(lambda x: pd.Series(x).rolling(self.window).mean(), price)
        self.std  = self.I(lambda x: pd.Series(x).rolling(self.window).std(),  price)

        # Guard against division by zero (std == 0)
        def zscore(x, m, s):
            s_safe = np.where((s == 0) | np.isnan(s), np.nan, s)
            return (x - m) / s_safe

        self.z = self.I(zscore, price, self.mean, self.std)

    def next(self):
        z_now = self.z[-1]

        # If we don't have enough data yet, skip
        if np.isnan(z_now):
            return

        # ENTRY: long only
        if not self.position:
            if z_now <= -self.z_entry:
                self.buy()

        # EXIT: close long when z reverts toward zero
        else:
            if self.position.is_long and z_now > -self.z_exit:
                self.position.close()
