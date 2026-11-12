"""Optional local spend guard based only on this app's recorded model usage."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC
from typing import Any

from app.config import Settings
from app.store import Store, iso, utcnow

logger = logging.getLogger(__name__)

OK = "ok"
PAUSED_FULL = "paused_full"
STOPPED = "stopped"


@dataclass(frozen=True)
class BudgetStatus:
    state: str = OK
    source: str = "local"
    spend_usd: float | None = None
    max_budget_usd: float | None = None
    fraction: float | None = None
    checked_at: str | None = None

    @property
    def allows_full(self) -> bool:
        return self.state == OK

    @property
    def allows_any(self) -> bool:
        return self.state != STOPPED

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.fraction is not None:
            data["fraction"] = round(self.fraction, 4)
        return data


def classify(fraction: float | None, pause: float, stop: float) -> str:
    if fraction is None:
        return OK
    if fraction >= stop:
        return STOPPED
    if fraction >= pause:
        return PAUSED_FULL
    return OK


class BudgetGuard:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.status = BudgetStatus()

    async def check(self) -> BudgetStatus:
        now = utcnow().astimezone(UTC)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        spend = float(self.store.cost_since(month_start)["cost"])
        budget = self.settings.LLM_MONTHLY_BUDGET_USD or None
        fraction = spend / budget if budget else None
        state = classify(
            fraction, self.settings.BUDGET_PAUSE_FRACTION, self.settings.BUDGET_STOP_FRACTION
        )
        if state != self.status.state:
            log = logger.warning if state != OK else logger.info
            log("Local spend guard: %s (spend $%.4f, limit %s)", state, spend, budget)
        self.status = BudgetStatus(
            state=state,
            source="local",
            spend_usd=round(spend, 4),
            max_budget_usd=budget,
            fraction=fraction,
            checked_at=iso(now),
        )
        return self.status
