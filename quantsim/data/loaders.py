from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from quantsim.engine.event_queue import EventQueue, MarketEvent

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_yfinance_ohlcv(
    symbols: list[str],
    start: str,
    end: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> dict[str, pd.DataFrame]:
    """Download (or read from local cache) daily OHLCV bars for each symbol."""
    import yfinance as yf

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        cache_path = cache_dir / f"{symbol}_{start}_{end}.csv"
        if cache_path.exists():
            frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frame = raw.rename(columns=str.lower)[REQUIRED_COLUMNS]
            frame.to_csv(cache_path)
        frames[symbol] = frame[REQUIRED_COLUMNS]
    return frames


def align_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Reindex all symbols to the union of trading dates, forward-filling gaps."""
    if not frames:
        return {}
    union_index = sorted(set().union(*(frame.index for frame in frames.values())))
    return {symbol: frame.reindex(union_index).ffill().dropna() for symbol, frame in frames.items()}


class HistoricDataHandler:
    """Drives an EventQueue with historical OHLCV bars in chronological order.

    Implements `get_latest_bars` (Strategy lookup, excludes the in-flight bar)
    and `get_next_bar` (ExecutionHandler lookup for next-open fills).
    """

    def __init__(self, frames: dict[str, pd.DataFrame], event_queue: EventQueue) -> None:
        self.frames = frames
        self.event_queue = event_queue
        self.symbols = list(frames.keys())
        self._push_cursor: dict[str, int] = {symbol: 0 for symbol in self.symbols}
        self._current_time: dict[str, datetime] = {}
        self.continue_backtest = True

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        frame = self.frames[symbol]
        current_time = self._current_time.get(symbol)
        if current_time is None:
            return frame.iloc[0:0]
        history = frame.loc[frame.index < current_time]
        return history.iloc[-n:] if n > 0 else history.iloc[0:0]

    def get_next_bar(self, symbol: str, after: datetime) -> MarketEvent | None:
        frame = self.frames[symbol]
        future = frame.index[frame.index > after]
        if future.empty:
            return None
        return self._row_to_event(symbol, future[0])

    def update_bars(self) -> None:
        """Advance one time step for every symbol, pushing MarketEvents onto the queue."""
        pushed = False
        for symbol in self.symbols:
            frame = self.frames[symbol]
            idx = self._push_cursor[symbol]
            if idx >= len(frame):
                continue
            self.event_queue.push(self._row_to_event(symbol, frame.index[idx]))
            self._push_cursor[symbol] = idx + 1
            pushed = True

        if not pushed:
            self.continue_backtest = False

    def mark_current(self, event: MarketEvent) -> None:
        """Record that `event` is now the in-flight bar for its symbol, so
        `get_latest_bars` excludes it (and includes it once processing moves
        on to a later bar)."""
        self._current_time[event.symbol] = event.timestamp

    def _row_to_event(self, symbol: str, ts: datetime) -> MarketEvent:
        row = self.frames[symbol].loc[ts]
        return MarketEvent(
            timestamp=ts,
            symbol=symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
