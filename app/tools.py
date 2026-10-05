"""Tools the agent can call.

Each tool receives a ToolContext holding the *authenticated* employee. The employee identity is never
taken from the LLM's arguments, so a prompt like "show me Rahul's leave balance" cannot leak data.
"""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

from . import db, rag
from .leave_rules import (LEAVE_RULES, check_eligibility, el_credited_months, entitlement, months_of_service,
                          on_probation, working_days, EL_MONTHLY_ACCRUAL)


class ToolError(Exception):
    """Raised for bad input; the message is returned to the LLM so it can correct itself."""


@dataclass
class ToolContext:
    employee: dict
    today: date


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]


TOOLS: dict[str, Tool] = {}
LEAVE_TYPE_PARAM = {"type": "string", "enum": list(LEAVE_RULES),
                    "description": "EL=Earned, CL=Casual, SL=Sick, PTL=Parental (secondary caregiver), BL=Bereavement"}
DATE_PARAM = {"type": "string", "description": "Date in YYYY-MM-DD format"}


def tool(description: str, properties: dict | None = None, required: list[str] | None = None):
    def register(fn):
        TOOLS[fn.__name__] = Tool(fn.__name__, description,
                                  {"type": "object", "properties": properties or {}, "required": required or []}, fn)
        return fn
    return register


def ollama_tool_schemas() -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.parameters}} for t in TOOLS.values()]


def execute_tool(name: str, arguments: dict, ctx: ToolContext) -> dict:
    tool_ = TOOLS.get(name)
    if not tool_:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}"}
    allowed = tool_.parameters["properties"]
    args = {k: v for k, v in (arguments or {}).items() if k in allowed and v not in (None, "")}
    missing = [r for r in tool_.parameters["required"] if r not in args]
    if missing:
        return {"error": f"Missing required argument(s): {', '.join(missing)}"}
    try:
        result = tool_.fn(ctx, **args)
    except ToolError as exc:
        return {"error": str(exc)}
    if name != "search_hr_policy":  # make it explicit whose data this is
        result = {"data_for_employee": f"{ctx.employee['full_name']} ({ctx.employee['id']})", **result}
    return result


# --- helpers -----------------------------------------------------------------

DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%d/%m/%Y", "%d-%m-%Y")


def _date(value: str, field: str) -> date:
    """Parse ISO dates, plus the human formats small models sometimes echo back (e.g. '12 Oct 2026')."""
    text = str(value).strip().replace(",", "")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ToolError(f"'{field}' must be a date in YYYY-MM-DD format, got '{value}'.")


def _bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"}


def _rule(code: str):
    rule = LEAVE_RULES.get(str(code).upper())
    if not rule:
        raise ToolError(f"Unknown leave type '{code}'. Use one of: {', '.join(LEAVE_RULES)}.")
    return rule


def _balance(employee: dict, code: str, as_of: date) -> dict:
    rule = LEAVE_RULES[code]
    granted, basis = entitlement(employee, rule, as_of)
    carried = db.get_carry_forward(employee["id"], as_of.year, code)
    used = db.sum_leave_days(employee["id"], as_of.year, code, "approved")
    pending = db.sum_leave_days(employee["id"], as_of.year, code, "pending")
    return {"leave_type": code, "name": rule.name, "granted_this_year": granted, "basis": basis,
            "carried_forward": carried, "used_approved": used, "pending_approval": pending,
            "available": granted + carried - used - pending}


# --- tools -------------------------------------------------------------------

@tool("Search the company's HR policy documents (leave, attendance, work-from-home, holidays, conduct, etc.). "
      "Use for any question about rules, eligibility criteria, procedures or definitions.",
      {"query": {"type": "string", "description": "A focused search query, e.g. 'earned leave carry forward limit'"}},
      ["query"])
def search_hr_policy(ctx: ToolContext, query: str) -> dict:
    results = rag.search_policies(query)
    if not results:
        return {"results": [], "message": "No relevant policy text found."}
    return {"results": results}


@tool("Get the logged-in employee's profile: department, designation, manager, join date, length of service, "
      "employment type and probation status.")
def get_my_profile(ctx: ToolContext) -> dict:
    emp = ctx.employee
    manager = db.get_employee(emp["manager_id"]) if emp["manager_id"] else None
    service = months_of_service(date.fromisoformat(emp["join_date"]), ctx.today)
    return {
        "employee_id": emp["id"], "name": emp["full_name"], "department": emp["department"],
        "designation": emp["designation"], "manager": manager["full_name"] if manager else None,
        "location": emp["location"], "employment_type": emp["employment_type"], "join_date": emp["join_date"],
        "service": f"{service // 12} year(s) {service % 12} month(s)",
        "on_probation": on_probation(emp, ctx.today), "probation_end_date": emp["probation_end_date"],
    }


@tool("Get the logged-in employee's current leave balance today (granted, carried forward, used, pending, "
      "available). Omit leave_type to get all leave types. Not for year-end lapse questions.",
      {"leave_type": LEAVE_TYPE_PARAM})
def get_leave_balance(ctx: ToolContext, leave_type: str | None = None) -> dict:
    codes = [_rule(leave_type).code] if leave_type else list(LEAVE_RULES)
    return {"as_of": ctx.today.isoformat(), "leave_year": f"Jan-Dec {ctx.today.year}",
            "balances": [_balance(ctx.employee, c, ctx.today) for c in codes]}


@tool("List the logged-in employee's leave requests.",
      {"year": {"type": "integer", "description": "Leave year, defaults to the current year"},
       "status": {"type": "string", "enum": ["pending", "approved", "rejected", "cancelled"]}})
def get_leave_history(ctx: ToolContext, year: int | None = None, status: str | None = None) -> dict:
    rows = db.get_leave_requests(ctx.employee["id"], int(year or ctx.today.year), status)
    return {"requests": [{k: r[k] for k in ("id", "leave_type", "start_date", "end_date", "days", "status", "reason")}
                         for r in rows]}


@tool("Count working days between two dates (inclusive), excluding weekends and company holidays.",
      {"start_date": DATE_PARAM, "end_date": DATE_PARAM}, ["start_date", "end_date"])
def calculate_working_days(ctx: ToolContext, start_date: str, end_date: str) -> dict:
    start, end = _date(start_date, "start_date"), _date(end_date, "end_date")
    if end < start:
        raise ToolError("end_date is before start_date.")
    holidays = {**db.get_holidays(start.year), **db.get_holidays(end.year)}
    days, excluded = working_days(start, end, holidays)
    return {"start_date": start_date, "end_date": end_date, "calendar_days": (end - start).days + 1,
            "working_days": len(days), "excluded": excluded}


@tool("Check whether the logged-in employee can take a specific leave for a date range: validates balance, "
      "probation, notice period, maximum consecutive days, overlaps, and counts working days.",
      {"leave_type": LEAVE_TYPE_PARAM, "start_date": DATE_PARAM, "end_date": DATE_PARAM,
       "half_day": {"type": "boolean", "description": "True for a half-day (start_date must equal end_date)"}},
      ["leave_type", "start_date", "end_date"])
def check_leave_eligibility(ctx: ToolContext, leave_type: str, start_date: str, end_date: str,
                            half_day: bool = False) -> dict:
    rule = _rule(leave_type)
    start, end = _date(start_date, "start_date"), _date(end_date, "end_date")
    holidays = {**db.get_holidays(start.year), **db.get_holidays(end.year)}
    # Balance is evaluated as of the leave start (EL keeps accruing until then).
    available = _balance(ctx.employee, rule.code, max(start, ctx.today))["available"]
    return check_eligibility(employee=ctx.employee, rule=rule, start=start, end=end, half_day=_bool(half_day),
                             available=available, holidays=holidays, today=ctx.today,
                             overlapping=db.get_overlapping_requests(ctx.employee["id"], start, end))


@tool("Use for ANY question about leave that will lapse, expire, be lost or be carried forward at year end. "
      "Projects the year-end position including EL still to be accrued and the carry-forward cap.")
def project_year_end_balance(ctx: ToolContext) -> dict:
    emp, year = ctx.employee, ctx.today.year
    join = date.fromisoformat(emp["join_date"])
    projections = []
    for code, rule in LEAVE_RULES.items():
        if rule.accrual == "per_event":
            continue
        bal = _balance(emp, code, ctx.today)
        if rule.accrual == "monthly":
            full_year = el_credited_months(join, year, 12) * EL_MONTHLY_ACCRUAL \
                if emp["employment_type"] in rule.employment_types else 0
            future_accrual = full_year - bal["granted_this_year"]
        else:
            future_accrual = 0
        year_end = bal["available"] + future_accrual
        carry = min(max(year_end, 0), rule.carry_forward_max)
        projections.append({"leave_type": code, "available_now": bal["available"],
                            "expected_further_accrual": future_accrual, "projected_year_end_balance": year_end,
                            "will_carry_forward": carry, "will_lapse": max(year_end - carry, 0),
                            "carry_forward_limit": rule.carry_forward_max})
    return {"leave_year": year, "assumes": "no further leave is taken apart from pending requests",
            "projections": projections}


@tool("List company holidays.",
      {"year": {"type": "integer", "description": "Defaults to the current year"},
       "upcoming_only": {"type": "boolean", "description": "Only holidays from today onwards (default true)"}})
def get_holidays(ctx: ToolContext, year: int | None = None, upcoming_only: bool = True) -> dict:
    holidays = db.get_holidays(int(year or ctx.today.year))
    if _bool(upcoming_only):
        holidays = {d: n for d, n in holidays.items() if d >= ctx.today}
    return {"holidays": [{"date": d.isoformat(), "day": d.strftime("%A"), "name": n} for d, n in holidays.items()]}


@tool("Submit a leave request for the logged-in employee (status becomes 'pending' manager approval). "
      "ONLY call after the employee has explicitly confirmed leave type, dates and reason.",
      {"leave_type": LEAVE_TYPE_PARAM, "start_date": DATE_PARAM, "end_date": DATE_PARAM,
       "reason": {"type": "string"},
       "half_day": {"type": "boolean"}},
      ["leave_type", "start_date", "end_date", "reason"])
def apply_leave(ctx: ToolContext, leave_type: str, start_date: str, end_date: str, reason: str,
                half_day: bool = False) -> dict:
    check = check_leave_eligibility(ctx, leave_type, start_date, end_date, half_day)
    if not check["eligible"]:
        return {"submitted": False, "issues": check["issues"]}
    request_id = db.insert_leave_request(ctx.employee["id"], check["leave_type"], _date(start_date, "start_date"),
                                         _date(end_date, "end_date"), check["days_requested"], _bool(half_day),
                                         reason)
    return {"submitted": True, "request_id": request_id, "status": "pending",
            "days": check["days_requested"], "balance_after": check["balance_after"], "notes": check["notes"]}
