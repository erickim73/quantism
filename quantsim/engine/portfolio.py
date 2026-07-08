from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from quantsim.engine.event_queue import FillEvent


@dataclass
class Position:
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0


class Portfolio:
    """Tracks cash, per-symbol positions, and realized/unrealized P&L from fills."""

    def __init__(self, initial_cash: float, symbols: list[str]) -> None:
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, Position] = {symbol: Position() for symbol in symbols}
        self.equity_curve: list[tuple[datetime, float]] = []

    def update_fill(self, fill: FillEvent) -> float:
        """Apply a fill to cash/positions and return the realized P&L delta it caused."""
        position = self.positions.setdefault(fill.symbol, Position())
        signed_qty = fill.quantity if fill.direction == "BUY" else -fill.quantity
        gross = fill.fill_price * fill.quantity

        if fill.direction == "BUY":
            self.cash -= gross + fill.commission
        else:
            self.cash += gross - fill.commission

        prev_qty = position.quantity
        new_qty = prev_qty + signed_qty
        is_reducing = prev_qty != 0 and (prev_qty > 0) != (signed_qty > 0)
        realized_pnl_delta = 0.0

        if is_reducing:
            closed_qty = min(abs(prev_qty), abs(signed_qty))
            direction_sign = 1 if prev_qty > 0 else -1
            realized_pnl_delta = closed_qty * direction_sign * (fill.fill_price - position.avg_price)
            position.realized_pnl += realized_pnl_delta
            if abs(signed_qty) > abs(prev_qty):
                # Position flipped sign: the excess opens a new leg at the fill price.
                position.avg_price = fill.fill_price
        else:
            total_cost = position.avg_price * abs(prev_qty) + fill.fill_price * abs(signed_qty)
            position.avg_price = total_cost / abs(new_qty) if new_qty != 0 else 0.0

        position.quantity = new_qty
        if new_qty == 0:
            position.avg_price = 0.0

        return realized_pnl_delta

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(pos.quantity * prices.get(symbol, 0.0) for symbol, pos in self.positions.items())

    def total_equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.quantity == 0:
                continue
            price = prices.get(symbol, pos.avg_price)
            total += pos.quantity * (price - pos.avg_price)
        return total

    def realized_pnl(self) -> float:
        return sum(pos.realized_pnl for pos in self.positions.values())

    def record_equity(self, timestamp: datetime, prices: dict[str, float]) -> None:
        self.equity_curve.append((timestamp, self.total_equity(prices)))
