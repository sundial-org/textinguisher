"""Process-wide compile semaphore and cookbook Message helpers shared by fix_env (the RL env)."""
import asyncio
import os

COMPILE_SLOTS = int(os.environ.get("COMPILE_SLOTS", "6"))
_compile_sem: asyncio.Semaphore | None = None


def compile_slot() -> asyncio.Semaphore:
    """Cap on concurrent latexmk runs in this process (COMPILE_SLOTS, default 6)."""
    global _compile_sem
    if _compile_sem is None:
        _compile_sem = asyncio.Semaphore(COMPILE_SLOTS)
    return _compile_sem


def _text(message) -> str:
    """Text of a cookbook Message (str or list of parts)."""
    c = message.get("content", "") if isinstance(message, dict) else message
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return c or ""
