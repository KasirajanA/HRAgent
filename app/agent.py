"""The HR agent: a ReAct-style loop where the Ollama model decides which tools to call until it can answer.

Performance notes (CPU inference):
- The system prompt and tool schemas (~1,300 tokens) are identical for every user and every day, so Ollama can
  reuse their KV cache between requests. Anything per-user/per-day goes into the *latest user message* instead.
  (The qwen2.5 template renders the system prompt before the tool block, so a varying system prompt would force
  the whole tool block to be re-processed on every request.)
- warm_up() fills that cache when the server starts; keep_alive keeps the model in memory between questions.
- run_agent_stream() streams tool activity and answer tokens so the UI shows progress immediately.
"""
import json
import threading
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Iterator

import ollama

from . import config, db
from .rag import ollama_client
from .tools import ToolContext, execute_tool, ollama_tool_schemas

SYSTEM_PROMPT = f"""You are "HR Assist", the HR chat agent for {config.COMPANY_NAME} employees.
Each user message starts with a [Context] block (today's date, the employee, a calendar). Use it to resolve
relative dates like "next Monday" into YYYY-MM-DD before calling tools. The leave year runs January to December.

## How to work
0. Every question needs fresh tool results - earlier answers in the chat may be outdated. Never reuse numbers from
   previous turns without calling the tool again.
1. Policy / rules / procedure questions: call `search_hr_policy` and answer ONLY from the returned excerpts.
   Mention the source document name. If the excerpts don't cover it, say so and suggest contacting HR ({config.HR_CONTACT}).
2. Personal data and numbers (balances, history, profile, holidays, working days, eligibility, projections):
   ALWAYS call the matching tool. Never guess numbers or do date arithmetic yourself.
3. "Can I take leave from X to Y?" -> `check_leave_eligibility`. "How many days is X to Y?" -> `calculate_working_days`.
   "How many leaves will lapse / carry forward?" -> `project_year_end_balance`.
   Combine tools when needed (e.g. eligibility + policy search to explain a rule).
4. Only call `apply_leave` when the employee explicitly asked to apply AND confirmed leave type, dates and reason.
   Otherwise summarise the request (including the eligibility check) and ask them to confirm.
5. Tools only return data for the logged-in employee. If asked about ANOTHER person's leave, profile or any personal
   data, do not call tools - politely refuse. Also refuse anything about salaries, bonuses, equity, insurance/medical
   claims or performance/ethics complaints, and refer them to HR.
6. If a tool returns an "error", fix the arguments and retry, or ask the user for the missing detail.
7. Be concise, friendly and professional. Only mention the figures that answer the question. Use short bullet
   points and format dates like 05 Oct 2026 in replies (tools always take YYYY-MM-DD)."""

GROUNDING_NUDGE = ("You answered without calling any tool. If the answer depends on HR policy or on the employee's "
                   "data, call the appropriate tool now. If no tool applies (greeting, refusal, clarification), "
                   "reply with your final answer again.")


def _context_block(employee: dict, today: date) -> str:
    calendar = ", ".join(f"{d.strftime('%a')} {d.isoformat()}" for d in (today + timedelta(days=i) for i in range(14)))
    return (f"[Context] Today: {today.strftime('%A')} {today.isoformat()}. Employee: {employee['full_name']} "
            f"({employee['id']}), {employee['designation']}, {employee['department']}. Next 14 days: {calendar}.")


class ConversationStore:
    """In-memory chat history per login session (user + assistant turns only, trimmed)."""

    def __init__(self):
        self._data: dict[str, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._data[session_id][-config.MAX_HISTORY_MESSAGES:])

    def append(self, session_id: str, *messages: dict) -> None:
        with self._lock:
            history = self._data[session_id]
            history.extend(messages)
            del history[:-config.MAX_HISTORY_MESSAGES]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


conversations = ConversationStore()


def _chat(messages: list[dict], tools: bool = True, stream: bool = False):
    return ollama_client().chat(
        model=config.CHAT_MODEL, messages=messages, tools=ollama_tool_schemas() if tools else None, stream=stream,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        options={"temperature": config.CHAT_TEMPERATURE, "num_ctx": config.NUM_CTX})


def warm_up() -> None:
    """Load the chat model and pre-compute the KV cache for the shared system prompt + tool schemas."""
    started = time.perf_counter()
    try:
        ollama_client().chat(model=config.CHAT_MODEL, tools=ollama_tool_schemas(), keep_alive=config.OLLAMA_KEEP_ALIVE,
                             messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "hi"}],
                             options={"temperature": 0, "num_ctx": config.NUM_CTX, "num_predict": 1})
        print(f"[agent] warmed up {config.CHAT_MODEL} in {time.perf_counter() - started:.0f}s")
    except Exception as exc:
        print(f"[agent] warm-up skipped: {exc}")


def run_agent_stream(session_id: str, employee: dict, user_message: str) -> Iterator[dict]:
    """Answer one user message, yielding events as they happen:
       {"type": "status", "content"}       what the agent is doing right now
       {"type": "step", "step"}            a finished tool call / thought / guardrail (for the trace panel)
       {"type": "token", "content"}        a piece of the final answer
       {"type": "done", "answer", "steps"} or {"type": "error", "answer", "steps"}
    """
    steps: list[dict] = []

    # Deterministic privacy guard: never let the model reason about another employee's data.
    others = db.find_other_employees_mentioned(user_message, employee["id"])
    if others:
        answer = (f"Sorry, I can only share HR information about you. I can't look up details for "
                  f"{', '.join(others)}. For questions about another employee, please contact HR ({config.HR_CONTACT}).")
        steps.append({"type": "guardrail", "content": f"Blocked: mentions {', '.join(others)}"})
        conversations.append(session_id, {"role": "user", "content": user_message},
                             {"role": "assistant", "content": answer})
        yield {"type": "step", "step": steps[-1]}
        yield {"type": "token", "content": answer}
        yield {"type": "done", "answer": answer, "steps": steps}
        return

    today = date.today()
    ctx = ToolContext(employee=employee, today=today)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                *conversations.get(session_id),
                {"role": "user", "content": f"{_context_block(employee, today)}\n\n{user_message}"}]
    answer = None
    nudged = False

    try:
        for step_no in range(config.MAX_AGENT_STEPS):
            used_tools = any(s["type"] == "tool" for s in steps)
            # Stream answer tokens only when this reply can't be discarded by the grounding nudge.
            live = used_tools or nudged or not config.GROUNDING_NUDGE
            yield {"type": "status", "content": "Thinking..." if step_no == 0 else "Reading tool results..."}

            content, tool_calls = "", []
            for chunk in _chat(messages, stream=True):
                if chunk.message.tool_calls:
                    tool_calls += chunk.message.tool_calls
                if chunk.message.content:
                    content += chunk.message.content
                    if live and not tool_calls:
                        yield {"type": "token", "content": chunk.message.content}

            if not tool_calls:
                if not live:
                    # Small models often answer from memory; give them one chance to ground the answer.
                    nudged = True
                    steps.append({"type": "thought", "content": "(no tool used - asking the model to verify)"})
                    yield {"type": "step", "step": steps[-1]}
                    messages += [{"role": "assistant", "content": content},
                                 {"role": "user", "content": GROUNDING_NUDGE}]
                    continue
                answer = content
                break

            messages.append({"role": "assistant", "content": content,
                             "tool_calls": [{"function": {"name": c.function.name,
                                                          "arguments": dict(c.function.arguments or {})}}
                                            for c in tool_calls]})
            if content.strip():
                steps.append({"type": "thought", "content": content})
                yield {"type": "step", "step": steps[-1]}
            for call in tool_calls:
                name, args = call.function.name, dict(call.function.arguments or {})
                yield {"type": "status", "content": f"Calling {name}..."}
                started = time.perf_counter()
                result = execute_tool(name, args, ctx)
                steps.append({"type": "tool", "tool": name, "arguments": args, "result": result,
                              "ms": round((time.perf_counter() - started) * 1000)})
                yield {"type": "step", "step": steps[-1]}
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(result, default=str, separators=(",", ":"))})

        if answer is None:  # step budget exhausted: force a final answer without tools
            yield {"type": "status", "content": "Summarising..."}
            messages.append({"role": "user", "content": "Summarise what you found so far and answer now."})
            answer = ""
            for chunk in _chat(messages, tools=False, stream=True):
                answer += chunk.message.content or ""
                yield {"type": "token", "content": chunk.message.content or ""}
    except ollama.ResponseError as exc:
        yield {"type": "error", "steps": steps,
               "answer": f"The language model returned an error: {exc.error}. "
                         f"Is the model '{config.CHAT_MODEL}' pulled in Ollama?"}
        return
    except Exception as exc:  # connection problems with Ollama / Qdrant
        yield {"type": "error", "steps": steps,
               "answer": f"Sorry, a backend service is unavailable ({type(exc).__name__}: {exc})."}
        return

    conversations.append(session_id, {"role": "user", "content": user_message},
                         {"role": "assistant", "content": answer})
    yield {"type": "done", "answer": answer, "steps": steps}


def run_agent(session_id: str, employee: dict, user_message: str) -> dict:
    """Non-streaming wrapper: returns {"answer", "steps"[, "error"]}."""
    for event in run_agent_stream(session_id, employee, user_message):
        if event["type"] == "done":
            return {"answer": event["answer"], "steps": event["steps"]}
        if event["type"] == "error":
            return {"answer": event["answer"], "steps": event["steps"], "error": True}
    return {"answer": "", "steps": [], "error": True}
