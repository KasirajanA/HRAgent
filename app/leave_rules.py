"""Leave policy rules encoded as code.

These mirror data/policies/leave_policy.md. The LLM reads the document (via RAG) to *explain* policy,
but all numbers (entitlements, accruals, working days, eligibility) are computed here so the
answers are deterministic and never left to the model's arithmetic.
"""
import math
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

EL_MONTHLY_ACCRUAL = 1.5
MIN_DAYS_ON_ROLLS_FOR_ACCRUAL = 15
INTERN_ENTITLEMENT = {"CL": 3, "SL": 6}


@dataclass(frozen=True)
class LeaveRule:
    code: str
    name: str
    annual_days: float
    accrual: str                            # "monthly" | "prorated_upfront" | "per_event"
    carry_forward_max: float = 0
    max_consecutive: float | None = None    # working days per single request
    notice_days: int = 0                    # advance notice required...
    notice_applies_above: float = 0         # ...when the request is longer than this many days
    allowed_in_probation: bool = True
    half_day_allowed: bool = False
    min_service_months: int = 0
    document_required_above: float | None = None
    backdate_days: int = 0                  # how far in the past a request may start
    employment_types: tuple[str, ...] = ("full_time", "contract")


LEAVE_RULES: dict[str, LeaveRule] = {r.code: r for r in [
    LeaveRule("EL", "Earned Leave", 18, "monthly", carry_forward_max=10, max_consecutive=15,
              notice_days=7, notice_applies_above=2, allowed_in_probation=False),
    LeaveRule("CL", "Casual Leave", 6, "prorated_upfront", max_consecutive=2, half_day_allowed=True,
              employment_types=("full_time", "contract", "intern")),
    LeaveRule("SL", "Sick Leave", 12, "prorated_upfront", half_day_allowed=True, document_required_above=2,
              backdate_days=7, employment_types=("full_time", "contract", "intern")),
    LeaveRule("PTL", "Parental Leave (secondary caregiver)", 10, "per_event", max_consecutive=10,
              notice_days=14, min_service_months=6),
    LeaveRule("BL", "Bereavement Leave", 3, "per_event", max_consecutive=3, backdate_days=7),
]}


def round_half(value: float) -> float:
    """Round to the nearest 0.5 day."""
    return math.floor(value * 2 + 0.5) / 2


def months_of_service(join: date, as_of: date) -> int:
    months = (as_of.year - join.year) * 12 + as_of.month - join.month
    if as_of.day < join.day:
        months -= 1
    return max(months, 0)


def on_probation(employee: dict, on: date) -> bool:
    return on <= date.fromisoformat(employee["probation_end_date"])


def el_credited_months(join: date, year: int, through_month: int) -> int:
    """Months 1..through_month of `year` in which the employee was on rolls for >= 15 days."""
    count = 0
    for month in range(1, through_month + 1):
        month_start = date(year, month, 1)
        month_end = date(year, month, monthrange(year, month)[1])
        if join > month_end:
            continue
        if (month_end - max(join, month_start)).days + 1 >= MIN_DAYS_ON_ROLLS_FOR_ACCRUAL:
            count += 1
    return count


def entitlement(employee: dict, rule: LeaveRule, as_of: date) -> tuple[float, str]:
    """Days granted for the leave year of `as_of` (excluding carry-forward), plus a human-readable basis."""
    if employee["employment_type"] not in rule.employment_types:
        return 0, f"Not applicable to {employee['employment_type']} employees"

    join = date.fromisoformat(employee["join_date"])
    annual = INTERN_ENTITLEMENT.get(rule.code, rule.annual_days) if employee["employment_type"] == "intern" \
        else rule.annual_days

    if rule.accrual == "monthly":
        # EL for a month is credited on the 1st of the following month.
        months = el_credited_months(join, as_of.year, as_of.month - 1)
        return months * EL_MONTHLY_ACCRUAL, f"{months} month(s) credited x {EL_MONTHLY_ACCRUAL} days"

    if rule.accrual == "prorated_upfront":
        if join < date(as_of.year, 1, 1):
            return annual, f"Full-year entitlement of {annual:g} days"
        months = 12 - join.month + (1 if join.day <= 15 else 0)
        return round_half(annual * months / 12), f"Prorated: {annual:g} x {months}/12 months (joined {join})"

    return annual, f"{annual:g} days per qualifying event"


def working_days(start: date, end: date, holidays: dict[date, str]) -> tuple[list[date], list[str]]:
    """Return (working dates, human-readable list of excluded dates) between start and end inclusive."""
    days, excluded = [], []
    current = start
    while current <= end:
        if current.weekday() >= 5:
            excluded.append(f"{current} ({current.strftime('%A')})")
        elif current in holidays:
            excluded.append(f"{current} (Holiday: {holidays[current]})")
        else:
            days.append(current)
        current += timedelta(days=1)
    return days, excluded


def check_eligibility(*, employee: dict, rule: LeaveRule, start: date, end: date, half_day: bool,
                      available: float, holidays: dict[date, str], today: date,
                      overlapping: list[dict]) -> dict:
    """Validate a prospective leave request against the policy. Returns issues (blocking) and notes."""
    issues: list[str] = []
    notes: list[str] = []

    if end < start:
        return {"eligible": False, "issues": ["End date is before start date."], "notes": [], "days_requested": 0}
    if start.year != end.year:
        issues.append("Leave cannot span two leave years (Jan-Dec). Split it into two requests.")

    dates, excluded = working_days(start, end, holidays)
    days = len(dates)
    if half_day:
        if not rule.half_day_allowed:
            issues.append(f"Half-day is not allowed for {rule.name}.")
        elif start != end:
            issues.append("A half-day request must have the same start and end date.")
        days = 0.5 if dates else 0
    if days == 0:
        issues.append("The selected range contains no working days (only weekends/holidays).")

    if employee["employment_type"] not in rule.employment_types:
        issues.append(f"{rule.name} is not available to {employee['employment_type']} employees.")
    if not rule.allowed_in_probation and on_probation(employee, start):
        issues.append(f"{rule.name} cannot be availed during probation "
                      f"(probation ends {employee['probation_end_date']}). It still accrues.")
    service = months_of_service(date.fromisoformat(employee["join_date"]), start)
    if service < rule.min_service_months:
        issues.append(f"{rule.name} requires {rule.min_service_months} months of service; "
                      f"you will have {service} month(s) on {start}.")

    if start < today - timedelta(days=rule.backdate_days):
        issues.append(f"{rule.name} cannot start more than {rule.backdate_days} day(s) in the past."
                      if rule.backdate_days else f"{rule.name} cannot be applied for past dates.")
    notice_given = (start - today).days
    if rule.notice_days and days > rule.notice_applies_above and notice_given < rule.notice_days:
        issues.append(f"{rule.name} longer than {rule.notice_applies_above:g} day(s) needs {rule.notice_days} days' "
                      f"advance notice; only {max(notice_given, 0)} day(s) given.")
    if rule.max_consecutive and days > rule.max_consecutive:
        issues.append(f"{rule.name} allows at most {rule.max_consecutive:g} working days per request.")

    if days > available:
        issues.append(f"Insufficient balance: {days:g} day(s) requested, {available:g} available. "
                      f"The shortfall of {days - available:g} day(s) would be Loss of Pay (needs manager approval).")
    if overlapping:
        ids = ", ".join(f"#{r['id']} ({r['status']})" for r in overlapping)
        issues.append(f"Overlaps with existing request(s): {ids}.")

    if rule.document_required_above and days > rule.document_required_above:
        notes.append(f"A medical certificate is required for {rule.name} longer than "
                     f"{rule.document_required_above:g} consecutive days.")
    if excluded:
        notes.append("Not counted (weekends/holidays): " + ", ".join(excluded))

    return {
        "eligible": not issues,
        "leave_type": rule.code,
        "days_requested": days,
        "balance_available": available,
        "balance_after": available - days if not issues else None,
        "issues": issues,
        "notes": notes,
    }
