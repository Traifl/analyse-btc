"""Tests for bot risk manager module."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.risk_manager import RiskManager
from bot.strategy import Signal
from bot.state import BotState


@pytest.fixture
def state(tmp_path):
    return BotState(initial_capital=10000, state_file=str(tmp_path / "test_state.json"))


@pytest.fixture
def rm():
    return RiskManager(
        max_risk_pct=0.02,
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
        cooldown_hours=4,
    )


def test_position_sizing(rm, state):
    qty = rm.calculate_position_size(10000, 50000)
    # risk = 10000 * 0.02 = 200, position_value = 200 / 0.03 = 6666.67
    assert qty == pytest.approx(6666.67 / 50000, rel=0.01)


def test_check_risk_buy_signal(rm, state):
    signal = Signal(1, 0.8, "combined")
    approved, qty = rm.check_risk(signal, 50000, state)
    assert approved is not None
    assert approved.direction == 1
    assert qty > 0


def test_check_risk_hold_signal(rm, state):
    signal = Signal(0, 0.5, "combined")
    approved, qty = rm.check_risk(signal, 50000, state)
    assert approved is None
    assert qty == 0.0


def test_check_risk_sell_no_position(rm, state):
    signal = Signal(-1, 0.8, "combined")
    approved, qty = rm.check_risk(signal, 50000, state)
    assert approved is None  # nothing to sell


def test_check_risk_sell_with_position(rm, state):
    state.open_position("long", 50000, 0.1)
    state.set("capital", 5000)  # remaining capital after buy
    signal = Signal(-1, 0.8, "combined")
    approved, qty = rm.check_risk(signal, 51000, state)
    assert approved is not None
    assert approved.direction == -1
    assert qty == 0.1


def test_stop_loss_triggered(rm, state):
    state.open_position("long", 50000, 0.1)
    # Price dropped 4% -> below 3% stop loss
    sl = rm.check_stop_loss_take_profit(state, 48000)
    assert sl is not None
    assert sl.name == "stop_loss"


def test_take_profit_triggered(rm, state):
    state.open_position("long", 50000, 0.1)
    # Price rose 6% -> above 5% take profit
    tp = rm.check_stop_loss_take_profit(state, 53000)
    assert tp is not None
    assert tp.name == "take_profit"


def test_no_sl_tp_within_range(rm, state):
    state.open_position("long", 50000, 0.1)
    result = rm.check_stop_loss_take_profit(state, 50500)
    assert result is None


def test_cooldown_blocks_trade(rm, state):
    # Add a recent trade
    state.add_trade("BUY", 50000, 0.1)
    assert not rm.check_cooldown(state)


def test_cooldown_allows_trade(rm, state):
    # Add an old trade
    old_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    state._state["trades"] = [{"time": old_time, "type": "SELL", "price": 50000, "quantity": 0.1, "pnl": 0}]
    assert rm.check_cooldown(state)


def test_buy_rejected_during_cooldown(rm, state):
    state.add_trade("SELL", 50000, 0.1)
    signal = Signal(1, 0.8, "combined")
    approved, qty = rm.check_risk(signal, 50000, state)
    assert approved is None  # cooldown blocks
