"""Runs inside hermes-lcm's host environment: reads JSON lines on stdin, answers on stdout.

Started by `hermes_lcm.py`, never imported by the kit, because it imports Hermes Agent's packages.
"""
import importlib.util
import json
import os
import re
import sys
import tempfile

plugin, window = sys.argv[1], int(sys.argv[2])
spec = importlib.util.spec_from_file_location("hermes_lcm", os.path.join(plugin, "__init__.py"),
                                              submodule_search_locations=[plugin])
module = importlib.util.module_from_spec(spec)
sys.modules["hermes_lcm"] = module
spec.loader.exec_module(module)

summary_tokens = [0]
if os.environ.get("EUNOE_HERMES_REAL_LLM") != "1":
    import agent.auxiliary_client as aux

    class _Reply:
        def __init__(self, text):
            self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]
            self.usage = None

    def _digest(**kw):
        text = json.dumps(kw.get("messages") or "")
        found = re.findall(r"\b[A-Z]{1,6}-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", text)
        out = "Earlier steps. Identifiers seen: " + ", ".join(dict.fromkeys(found))[:1200]
        summary_tokens[0] += len(out) // 4
        return _Reply(out)

    aux.call_llm = _digest

from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.engine import LCMEngine  # noqa: E402

os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="eunoe-hermes-")
engine = LCMEngine(LCMConfig.from_env() if hasattr(LCMConfig, "from_env") else LCMConfig())
engine.on_session_start("eunoe", hermes_home=os.environ["HERMES_HOME"], context_length=window * 2)


def answer(request: dict) -> dict:
    op = request.get("op")
    if op == "ping":
        return {"ok": True}
    if op == "compress":
        summary_tokens[0] = 0
        messages = engine.compress(request["messages"], current_tokens=request.get("tokens"), force=True)
        return {"status": engine.last_compression_status, "messages": messages, "summary_tokens": summary_tokens[0]}
    if op == "search":
        result = engine.handle_tool_call("lcm_grep", {"pattern": request.get("query", ""), "query": request.get("query", "")})
        return {"text": result if isinstance(result, str) else json.dumps(result)}
    return {"error": f"unknown op {op}"}


for line in sys.stdin:
    try:
        out = answer(json.loads(line))
    except Exception as exc:  # the parent reads a line per request, whatever happens
        out = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    sys.stdout.write(json.dumps(out, default=str) + "\n")
    sys.stdout.flush()
