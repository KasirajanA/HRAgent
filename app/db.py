"""SQLite HR database: schema, synthetic seed data and query helpers.

All seed data is fictional. Run `python -m app.db --reset` to rebuild it.
"""
import argparse
import hashlib
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

from . import config
from .leave_rules import working_days

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id                 TEXT PRIMARY KEY,
    username           TEXT UNIQUE NOT NULL,
    password_hash      TEXT NOT NULL,
    full_name          TEXT NOT NULL,
    email              TEXT NOT NULL,
    department         TEXT NOT NULL,
    designation        TEXT NOT NULL,
    manager_id         TEXT REFERENCES employees(id),
    join_date          TEXT NOT NULL,
    probation_end_date TEXT NOT NULL,
    employment_type    TEXT NOT NULL CHECK (employment_type IN ('full_time', 'contract', 'intern')),
    location           TEXT NOT NULL,
    role               TEXT NOT NULL DEFAULT 'employee' CHECK (role IN ('employee', 'hr_admin'))
);
CREATE TABLE IF NOT EXISTS leave_carry_forward (
    employee_id TEXT NOT NULL REFERENCES employees(id),
    year        INTEGER NOT NULL,
    leave_type  TEXT NOT NULL,
    days        REAL NOT NULL,
    PRIMARY KEY (employee_id, year, leave_type)
);
CREATE TABLE IF NOT EXISTS leave_requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL REFERENCES employees(id),
    leave_type  TEXT NOT NULL,
    start_date  TEXT NOT NULL,
    end_date    TEXT NOT NULL,
    days        REAL NOT NULL,
    half_day    INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled')),
    reason      TEXT,
    applied_on  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS holidays (
    holiday_date TEXT PRIMARY KEY,
    name         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(id),
    expires_at  TEXT NOT NULL
);
"""

# Must match data/policies/holiday_calendar_2026.md
HOLIDAYS_2026 = [
    ("2026-01-01", "New Year's Day"), ("2026-01-15", "Pongal"), ("2026-01-26", "Republic Day"),
    ("2026-03-04", "Holi"), ("2026-04-03", "Good Friday"), ("2026-05-01", "May Day"),
    ("2026-08-15", "Independence Day"), ("2026-09-14", "Ganesh Chaturthi"), ("2026-10-02", "Gandhi Jayanti"),
    ("2026-10-20", "Vijayadashami"), ("2026-11-09", "Diwali"), ("2026-12-25", "Christmas"),
]


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _rows(cursor) -> list[dict]:
    return [dict(r) for r in cursor.fetchall()]


# --- passwords & sessions ---------------------------------------------------

def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    _, salt, digest = stored.split("$")
    return secrets.compare_digest(hash_password(password, salt).split("$")[2], digest)


def create_session(employee_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=config.SESSION_TTL_HOURS)
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (datetime.now(timezone.utc).isoformat(),))
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?)", (token, employee_id, expires.isoformat()))
    return token


def get_session_employee(token: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT e.* FROM sessions s JOIN employees e ON e.id = s.employee_id "
            "WHERE s.token = ? AND s.expires_at > ?", (token, datetime.now(timezone.utc).isoformat())
        ).fetchone()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# --- employee & leave queries ------------------------------------------------

def get_employee_by_username(username: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM employees WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def get_employee(employee_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    return dict(row) if row else None


def find_other_employees_mentioned(text: str, self_id: str) -> list[str]:
    """Names/IDs of *other* employees mentioned in text (first name, full name or employee ID, whole words)."""
    lowered = f" {re.sub(r'[^a-z0-9]+', ' ', text.lower())} "
    with connect() as conn:
        rows = conn.execute("SELECT id, full_name FROM employees WHERE id != ?", (self_id,)).fetchall()
    return [r["full_name"] for r in rows
            if any(f" {term} " in lowered for term in (r["id"].lower(), r["full_name"].lower(),
                                                        r["full_name"].split()[0].lower()))]


def get_holidays(year: int) -> dict[date, str]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM holidays WHERE holiday_date LIKE ? ORDER BY holiday_date",
                            (f"{year}-%",)).fetchall()
    return {date.fromisoformat(r["holiday_date"]): r["name"] for r in rows}


def get_carry_forward(employee_id: str, year: int, leave_type: str) -> float:
    with connect() as conn:
        row = conn.execute("SELECT days FROM leave_carry_forward WHERE employee_id = ? AND year = ? "
                           "AND leave_type = ?", (employee_id, year, leave_type)).fetchone()
    return row["days"] if row else 0.0


def sum_leave_days(employee_id: str, year: int, leave_type: str, status: str) -> float:
    with connect() as conn:
        row = conn.execute("SELECT COALESCE(SUM(days), 0) AS total FROM leave_requests WHERE employee_id = ? "
                           "AND leave_type = ? AND status = ? AND start_date LIKE ?",
                           (employee_id, leave_type, status, f"{year}-%")).fetchone()
    return row["total"]


def get_leave_requests(employee_id: str, year: int | None = None, status: str | None = None) -> list[dict]:
    sql, params = "SELECT * FROM leave_requests WHERE employee_id = ?", [employee_id]
    if year:
        sql += " AND start_date LIKE ?"
        params.append(f"{year}-%")
    if status:
        sql += " AND status = ?"
        params.append(status)
    with connect() as conn:
        return _rows(conn.execute(sql + " ORDER BY start_date DESC", params))


def get_overlapping_requests(employee_id: str, start: date, end: date) -> list[dict]:
    with connect() as conn:
        return _rows(conn.execute(
            "SELECT * FROM leave_requests WHERE employee_id = ? AND status IN ('pending', 'approved') "
            "AND start_date <= ? AND end_date >= ?", (employee_id, end.isoformat(), start.isoformat())))


def insert_leave_request(employee_id: str, leave_type: str, start: date, end: date, days: float,
                         half_day: bool, reason: str, status: str = "pending") -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO leave_requests (employee_id, leave_type, start_date, end_date, days, half_day, status, "
            "reason, applied_on) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (employee_id, leave_type, start.isoformat(), end.isoformat(), days, int(half_day), status, reason,
             date.today().isoformat()))
        return cur.lastrowid


# --- setup -------------------------------------------------------------------

def init_db(reset: bool = False) -> None:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if reset and config.DB_PATH.exists():
        config.DB_PATH.unlink()
    with connect() as conn:
        conn.executescript(SCHEMA)
        empty = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 0
    if empty:
        _seed()


def _seed() -> None:
    """Insert fictional employees and leave history. Every demo user's password is `demo123`."""
    today = date.today()
    year = today.year
    pw = hash_password("demo123")
    probation_joiner = today - timedelta(days=50)
    employees = [
        # id, username, name, dept, designation, manager, join, probation_end, type, location, role
        ("E1003", "meera", "Meera Iyer", "Human Resources", "HR Manager", None,
         "2018-07-02", "2018-10-02", "full_time", "Chennai", "hr_admin"),
        ("E1001", "asha", "Asha Raman", "Engineering", "Senior Software Engineer", "E1003",
         "2021-03-15", "2021-06-15", "full_time", "Chennai", "employee"),
        ("E1002", "rahul", "Rahul Mehta", "Engineering", "Software Engineer", "E1003",
         probation_joiner.isoformat(), (probation_joiner + timedelta(days=91)).isoformat(), "full_time", "Bengaluru",
         "employee"),
        ("E1004", "arjun", "Arjun Das", "Quality Assurance", "QA Intern", "E1003",
         f"{year}-06-01", f"{year}-06-01", "intern", "Chennai", "employee"),
    ]
    holidays = {date.fromisoformat(d): n for d, n in HOLIDAYS_2026}

    def req(emp, code, start, end, status, reason, half=False):
        s, e = date.fromisoformat(start), date.fromisoformat(end)
        days = 0.5 if half else len(working_days(s, e, holidays)[0])
        return (emp, code, start, end, days, int(half), status, reason, min(s - timedelta(days=10), today).isoformat())

    pending_start = today + timedelta(days=21)
    while pending_start.weekday() >= 5:
        pending_start += timedelta(days=1)
    requests = [
        req("E1001", "EL", f"{year}-02-09", f"{year}-02-13", "approved", "Family trip"),
        req("E1001", "SL", f"{year}-04-20", f"{year}-04-21", "approved", "Fever"),
        req("E1001", "CL", f"{year}-06-12", f"{year}-06-12", "approved", "Personal work"),
        req("E1001", "CL", f"{year}-07-17", f"{year}-07-17", "approved", "Bank work", half=True),
        req("E1001", "EL", f"{year}-08-10", f"{year}-08-14", "approved", "Vacation"),
        req("E1001", "EL", pending_start.isoformat(), (pending_start + timedelta(days=1)).isoformat(),
            "pending", "Wedding in the family"),
        req("E1003", "EL", f"{year}-05-04", f"{year}-05-08", "approved", "Vacation"),
        req("E1003", "SL", f"{year}-09-02", f"{year}-09-03", "approved", "Medical"),
        req("E1004", "SL", f"{year}-07-08", f"{year}-07-08", "approved", "Cold"),
    ]

    with connect() as conn:
        conn.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            (eid, user, pw, name, f"{user}@acme.example", dept, desig, mgr, join, prob, etype, loc, role)
            for eid, user, name, dept, desig, mgr, join, prob, etype, loc, role in employees])
        conn.executemany("INSERT INTO leave_carry_forward VALUES (?, ?, ?, ?)",
                         [("E1001", year, "EL", 8), ("E1003", year, "EL", 10)])
        conn.executemany("INSERT INTO holidays VALUES (?, ?)", HOLIDAYS_2026)
        conn.executemany("INSERT INTO leave_requests (employee_id, leave_type, start_date, end_date, days, "
                         "half_day, status, reason, applied_on) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", requests)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and seed the HR SQLite database.")
    parser.add_argument("--reset", action="store_true", help="delete the existing database first")
    args = parser.parse_args()
    init_db(reset=args.reset)
    print(f"HR database ready at {config.DB_PATH}")
