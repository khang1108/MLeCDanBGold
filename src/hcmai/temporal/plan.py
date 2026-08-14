from __future__ import annotations

from dataclasses import dataclass

from hcmai.common.schemas.search import SearchFilters
from hcmai.temporal.models import QueryUnit, TemporalConstraint


@dataclass(frozen=True, slots=True)
class TemporalQueryPlan:
    units: tuple[QueryUnit, ...]
    constraints: tuple[TemporalConstraint, ...] = ()
    filters: SearchFilters | None = None

    def __post_init__(self) -> None:
        unit_ids = {unit.unit_id for unit in self.units}
        for constraint in self.constraints:
            referenced_ids = {constraint.left_unit_id, constraint.right_unit_id}
            unknown_ids = referenced_ids - unit_ids
            if unknown_ids:
                unknown_id = min(unknown_ids)
                raise ValueError(f"constraint references unknown unit: {unknown_id}")
