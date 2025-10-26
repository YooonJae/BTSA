import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover  # Close crosses ABOVE


def crossunder(a, b):
    """Close-to-close cross BELOW."""
    a, b = np.asarray(a), np.asarray(b)
    return (a[-2] >= b[-2]) and (a[-1] <= b[-1]) and (a[-1] < a[-2] or b[-1] > b[-2])


class TEST_1(Strategy):
    """
    MA High/Low Channel with width multiplier (LONG ONLY), close-to-close logic.

    Same rule for the selected band(s):
      - LONG entry : Close crosses UP through band
      - LONG exit  : Close crosses DOWN through band

    Configure which band(s) to use via `signal_bands`:
      - "lower", "middle", "upper" or any list/tuple combo, e.g. ("lower","upper")

    `trigger_logic`:
      - "any" = fire if ANY selected band crosses (default)
      - "all" = require ALL selected bands to cross on the bar
    """

    # --- Core params ---
    n = 180
    ma_type = "ema"            # "ema" or "sma"
    band_k = 10.0               # try 1.5–3.0 first; 10.0 is very wide and may rarely cross

    # --- Signal config ---
    signal_bands = ("lower", "middle", "upper")  # "lower" | "middle" | "upper" | combo
    trigger_logic = "any"      # "any" or "all"

    # --- Risk controls (optional) ---
    sl_pct = 0.0               # e.g., 0.02
    tp_pct = 0.0               # e.g., 0.03

    # --- Trade de-duplication ---
    min_bars_between_trades = 0

    # ---- internals ----
    _bands_map = None
    _signal_bands_list = None
    _last_signal_bar = -10**9

    def _ma(self, arr, n):
        s = pd.Series(arr)
        if self.ma_type.lower() == "ema":
            return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean().values
        return s.rolling(int(n), min_periods=int(n)).mean().values

    def _normalize_signal_bands(self):
        sb = self.signal_bands
        if isinstance(sb, str):
            sb = [sb]
        sb = [x.lower() for x in sb]
        valid = {"lower", "middle", "upper"}
        bad = [x for x in sb if x not in valid]
        if bad:
            raise ValueError(f"Invalid signal_bands {bad}; use any of {sorted(valid)}")
        # stable order
        order = ["lower", "middle", "upper"]
        return [b for b in order if b in sb]

    def init(self):
        H, L = self.data.High, self.data.Low
        ma_H = self._ma(H, self.n)
        ma_L = self._ma(L, self.n)

        mid = (ma_H + ma_L) / 2.0
        half = (ma_H - ma_L) / 2.0

        upper = mid + half * self.band_k
        lower = mid - half * self.band_k

        # Register as indicators (aligns lengths & plots)
        self.band_upper = self.I(lambda: upper)
        self.band_lower = self.I(lambda: lower)
        self.band_mid   = self.I(lambda: mid)

        self._bands_map = {
            "lower": self.band_lower,
            "middle": self.band_mid,
            "upper": self.band_upper,
        }
        self._signal_bands_list = self._normalize_signal_bands()

    def _apply_risk(self, entry_price):
        if self.sl_pct:
            self.position.set_stop_loss(entry_price * (1 - self.sl_pct))
        if self.tp_pct:
            self.position.set_take_profit(entry_price * (1 + self.tp_pct))

    # --- Cross evaluators over multiple bands (close-to-close only) ---
    def _cross_up(self, C):
        checks = []
        for name in self._signal_bands_list:
            line = self._bands_map[name]
            if np.isnan(line[-1]) or len(C) < 2:
                return False if self.trigger_logic == "all" else False
            checks.append(crossover(C, line))
        return any(checks) if self.trigger_logic == "any" else all(checks)

    def _cross_down(self, C):
        checks = []
        for name in self._signal_bands_list:
            line = self._bands_map[name]
            if np.isnan(line[-1]) or len(C) < 2:
                return False if self.trigger_logic == "all" else False
            checks.append(crossunder(C, line))
        return any(checks) if self.trigger_logic == "any" else all(checks)

    def next(self):
        C = self.data.Close
        i = len(C) - 1
        if len(C) < 2:
            return
        # NaN guard on selected bands
        if any(np.isnan(self._bands_map[name][-1]) for name in self._signal_bands_list):
            return

        # ---- EXIT (priority) ----
        if self.position and self._cross_down(C):
            self.position.close()
            return

        # ---- ENTRY ----
        if not self.position:
            if i - self._last_signal_bar < self.min_bars_between_trades:
                return
            if self._cross_up(C):
                self.buy()
                self._apply_risk(C[-1])
                self._last_signal_bar = i