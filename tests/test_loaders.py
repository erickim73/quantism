from datetime import datetime

import pandas as pd
import pytest

from quantsim.data.loaders import HistoricDataHandler, align_frames
from quantsim.engine.event_queue import EventQueue


def make_frame(dates, opens):
    index = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 1 for o in opens],
            "low": [o - 1 for o in opens],
            "close": [o + 0.5 for o in opens],
            "volume": [1000] * len(opens),
        },
        index=index,
    )


def test_align_frames_reindexes_to_union_of_dates_and_forward_fills():
    aapl = make_frame(["2024-01-01", "2024-01-03"], [100.0, 102.0])
    msft = make_frame(["2024-01-01", "2024-01-02", "2024-01-03"], [200.0, 201.0, 202.0])

    aligned = align_frames({"AAPL": aapl, "MSFT": msft})

    assert list(aligned["AAPL"].index) == list(aligned["MSFT"].index)
    # AAPL has no 2024-01-02 bar, so it should forward-fill from 2024-01-01.
    jan2 = pd.Timestamp("2024-01-02")
    assert aligned["AAPL"].loc[jan2, "open"] == pytest.approx(100.0)


def test_align_frames_empty_input_returns_empty_dict():
    assert align_frames({}) == {}


def test_update_bars_pushes_one_event_per_symbol_per_call():
    frames = {
        "AAPL": make_frame(["2024-01-01", "2024-01-02"], [100.0, 101.0]),
        "MSFT": make_frame(["2024-01-01", "2024-01-02"], [200.0, 201.0]),
    }
    queue = EventQueue()
    handler = HistoricDataHandler(frames, queue)

    handler.update_bars()

    assert len(queue) == 2
    assert handler.continue_backtest is True


def test_continue_backtest_becomes_false_once_all_bars_exhausted():
    frames = {"AAPL": make_frame(["2024-01-01"], [100.0])}
    queue = EventQueue()
    handler = HistoricDataHandler(frames, queue)

    handler.update_bars()  # pushes the only bar
    handler.update_bars()  # nothing left to push

    assert handler.continue_backtest is False


def test_get_latest_bars_excludes_the_marked_current_bar():
    frames = {"AAPL": make_frame(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 101.0, 102.0])}
    queue = EventQueue()
    handler = HistoricDataHandler(frames, queue)

    current_event = handler.get_next_bar("AAPL", after=datetime(2023, 12, 31))
    handler.mark_current(current_event)

    latest = handler.get_latest_bars("AAPL", n=5)

    assert len(latest) == 0  # no bars strictly before 2024-01-01


def test_get_latest_bars_returns_history_up_to_but_not_including_current():
    frames = {"AAPL": make_frame(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 101.0, 102.0])}
    queue = EventQueue()
    handler = HistoricDataHandler(frames, queue)

    handler.mark_current(handler.get_next_bar("AAPL", after=pd.Timestamp("2024-01-02")))  # marks 01-03

    latest = handler.get_latest_bars("AAPL", n=5)

    assert list(latest["open"]) == [100.0, 101.0]


def test_get_next_bar_returns_none_past_end_of_data():
    frames = {"AAPL": make_frame(["2024-01-01"], [100.0])}
    queue = EventQueue()
    handler = HistoricDataHandler(frames, queue)

    assert handler.get_next_bar("AAPL", after=pd.Timestamp("2024-01-01")) is None
