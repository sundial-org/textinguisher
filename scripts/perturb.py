"""Error injection: each mutation returns a break edit {type, search, replace} on the
original text, where both windows are line-anchored and unique, or None if inapplicable.
The fix target is the reverse edit. Non-failing perturbations are discarded by the generator."""
import random
import re

COMMON_ENVS = ["itemize", "enumerate", "figure", "table", "equation", "align", "center", "abstract"]


def _line_span(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, len(text) if end == -1 else end


def _unique_window(text: str, start: int, end: int) -> tuple[str, int, int] | None:
    """Grow [start,end) by whole lines until the window is unique in text (max 4 lines)."""
    for _ in range(4):
        window = text[start:end]
        if window.strip() and text.count(window) == 1:
            return window, start, end
        nxt = text.find("\n", end)
        if nxt == -1:
            return None
        end = nxt if nxt != end else _line_span(text, nxt + 1)[1]
        if end <= start:
            return None
    return None


def _window_edit(text: str, pos: int, mutate, mtype: str, rng) -> dict | None:
    ls, le = _line_span(text, pos)
    got = _unique_window(text, ls, le)
    if not got:
        return None
    search, s, e = got
    replace = mutate(search, pos - s)
    if replace is None or replace == search:
        return None
    if text.replace(search, replace, 1).count(replace) != 1:
        return None
    return {"type": mtype, "search": search, "replace": replace}


def _pick(matches, rng, in_preamble_ok=True, text="", preamble_end=0):
    matches = list(matches)
    if not in_preamble_ok:
        matches = [m for m in matches if m.start() > preamble_end]
    rng.shuffle(matches)
    return matches


def _preamble_end(text: str) -> int:
    m = re.search(r"\\begin\{document\}", text)
    return m.start() if m else 0


MACRO_RE = re.compile(r"\\([a-zA-Z]{4,})")
SKIP_MACROS = {"begin", "end", "documentclass", "usepackage", "newcommand", "renewcommand",
               "providecommand", "newenvironment", "input", "include", "bibliography",
               "bibliographystyle", "RequirePackage", "LoadClass", "ProvidesPackage"}


def typo_macro(text: str, rng: random.Random) -> dict | None:
    body = _preamble_end(text)
    for m in _pick(MACRO_RE.finditer(text), rng, False, text, body):
        name = m.group(1)
        if name in SKIP_MACROS:
            continue
        i = rng.randrange(len(name) - 1)
        bad = name[:i] + name[i + 1] + name[i] + name[i + 2:]

        def mut(win, off, name=name, bad=bad):
            return win[:off] + "\\" + bad + win[off + 1 + len(name):]

        edit = _window_edit(text, m.start(), mut, "typo_macro", rng)
        if edit:
            return edit
    return None


DOLLAR_RE = re.compile(r"(?<!\$)\$([^$\n]{2,80})\$(?!\$)")


def drop_dollar(text: str, rng: random.Random) -> dict | None:
    for m in _pick(DOLLAR_RE.finditer(text), rng):
        def mut(win, off, m=m):
            close = off + len(m.group(0)) - 1
            return win[:close] + win[close + 1:]

        edit = _window_edit(text, m.start(), mut, "drop_dollar", rng)
        if edit:
            return edit
    return None


BRACE_ARG_RE = re.compile(
    r"\\(textbf|textit|textsc|texttt|emph|underline|footnote|mbox|section|subsection|"
    r"subsubsection|paragraph|caption|title|author|thanks|mathrm|mathbf|text|label|cite|ref)"
    r"\{([^{}\n]{2,60})\}"
)


def drop_brace(text: str, rng: random.Random) -> dict | None:
    for m in _pick(BRACE_ARG_RE.finditer(text), rng):
        def mut(win, off, m=m):
            close = off + len(m.group(0)) - 1
            return win[:close] + win[close + 1:]

        edit = _window_edit(text, m.start(), mut, "drop_brace", rng)
        if edit:
            return edit
    return None


def extra_brace(text: str, rng: random.Random) -> dict | None:
    body = _preamble_end(text)
    lines_pos = [m.start() for m in re.finditer(r"[^\s\\%{}]\n", text) if m.start() > body]
    rng.shuffle(lines_pos)
    for pos in lines_pos[:20]:
        edit = _window_edit(text, pos, lambda win, off: win[:off + 1] + "}" + win[off + 1:], "extra_brace", rng)
        if edit:
            return edit
    return None


END_ENV_RE = re.compile(r"\\end\{([a-zA-Z*]+)\}")


def env_mismatch(text: str, rng: random.Random) -> dict | None:
    for m in _pick(END_ENV_RE.finditer(text), rng):
        name = m.group(1)
        if name == "document":
            continue
        other = rng.choice([e for e in COMMON_ENVS if e != name])

        def mut(win, off, m=m, other=other):
            return win[:off] + f"\\end{{{other}}}" + win[off + len(m.group(0)):]

        edit = _window_edit(text, m.start(), mut, "env_mismatch", rng)
        if edit:
            return edit
    return None


BEGIN_ENV_RE = re.compile(r"\\begin\{([a-zA-Z]{4,})\}")


def typo_begin_env(text: str, rng: random.Random) -> dict | None:
    for m in _pick(BEGIN_ENV_RE.finditer(text), rng):
        name = m.group(1)
        if name == "document":
            continue
        i = rng.randrange(len(name) - 1)
        bad = name[:i] + name[i + 1] + name[i] + name[i + 2:]

        def mut(win, off, m=m, bad=bad):
            return win[:off] + f"\\begin{{{bad}}}" + win[off + len(m.group(0)):]

        edit = _window_edit(text, m.start(), mut, "typo_begin_env", rng)
        if edit:
            return edit
    return None


USEPACKAGE_RE = re.compile(r"^[ \t]*\\usepackage(\[[^\]]*\])?\{[^}]+\}[^\n]*\n", re.M)


def drop_usepackage(text: str, rng: random.Random) -> dict | None:
    for m in _pick(USEPACKAGE_RE.finditer(text), rng):
        s, e = m.start(), m.end()
        got = _unique_window(text, s, _line_span(text, e)[1])
        if not got:
            continue
        search, ws, we = got
        replace = search[: s - ws] + search[e - ws:]
        if not replace.strip() or text.replace(search, replace, 1).count(replace) != 1:
            continue
        return {"type": "drop_usepackage", "search": search, "replace": replace}
    return None


def extra_amp(text: str, rng: random.Random) -> dict | None:
    body = _preamble_end(text)
    words = [m for m in re.finditer(r"(?<=[a-z]) (?=[a-z])", text) if m.start() > body]
    rng.shuffle(words)
    for m in words[:20]:
        edit = _window_edit(text, m.start(), lambda win, off: win[:off] + " &" + win[off:], "extra_amp", rng)
        if edit:
            return edit
    return None


DEF_RE = re.compile(r"^[ \t]*\\(?:newcommand|providecommand|DeclareMathOperator\*?)\{?\\[a-zA-Z]+\}?[^\n]*\n", re.M)


def drop_newcommand(text: str, rng: random.Random) -> dict | None:
    for m in _pick(DEF_RE.finditer(text), rng):
        cmd = re.search(r"\\(?:newcommand|providecommand|DeclareMathOperator\*?)\{?\\([a-zA-Z]+)", m.group(0))
        if not cmd or len(re.findall(r"\\" + cmd.group(1) + r"\b", text)) < 2:
            continue  # only drop definitions that are actually used later
        s, e = m.start(), m.end()
        got = _unique_window(text, s, _line_span(text, e)[1])
        if not got:
            continue
        search, ws, we = got
        replace = search[: s - ws] + search[e - ws:]
        if not replace.strip() or text.replace(search, replace, 1).count(replace) != 1:
            continue
        return {"type": "drop_newcommand", "search": search, "replace": replace}
    return None


NEWCMD_BODY_RE = re.compile(r"^[ \t]*\\(?:newcommand|renewcommand)\{?\\([a-zA-Z]+)\}?(?:\[\d+\])?\{(.+)\}[ \t]*$", re.M)


def at_distance(text: str, rng: random.Random) -> dict | None:
    """Corrupt a macro DEFINITION body; errors surface at every use site, far away."""
    for m in _pick(NEWCMD_BODY_RE.finditer(text), rng):
        name, body = m.group(1), m.group(2)
        if len(re.findall(r"\\" + name + r"\b", text)) < 3:
            continue  # definition + >=2 uses, so the distance effect is real
        inner = re.search(r"\\([a-zA-Z]{4,})", body)
        if not inner:
            continue
        i = rng.randrange(len(inner.group(1)) - 1)
        w = inner.group(1)
        bad = w[:i] + w[i + 1] + w[i] + w[i + 2:]

        def mut(win, off, m=m, w=w, bad=bad):
            rel = win.find("\\" + w, off)
            if rel == -1 or rel >= off + len(m.group(0)):
                return None
            return win[:rel] + "\\" + bad + win[rel + 1 + len(w):]

        edit = _window_edit(text, m.start(), mut, "at_distance", rng)
        if edit:
            return edit
    return None


def option_clash(text: str, rng: random.Random) -> dict | None:
    """Duplicate a \\usepackage with a clashing option -> 'Option clash for package'."""
    for m in _pick(USEPACKAGE_RE.finditer(text), rng):
        pkg = re.search(r"\{([^}]+)\}", m.group(0))
        if not pkg or "," in pkg.group(1):
            continue

        def mut(win, off, m=m, pkg=pkg):
            end = off + len(m.group(0).rstrip("\n"))
            return win[:end] + f"\n\\usepackage[draft]{{{pkg.group(1)}}}" + win[end:]

        edit = _window_edit(text, m.start(), mut, "option_clash", rng)
        if edit:
            return edit
    return None


SECTION_TITLE_RE = re.compile(r"\\(?:section|subsection)\{([^{}\n]{8,60})\}")


def fragile_cmd(text: str, rng: random.Random) -> dict | None:
    """Plant a fragile \\footnote inside a sectioning title (moving argument)."""
    if "\\tableofcontents" not in text:
        return None
    for m in _pick(SECTION_TITLE_RE.finditer(text), rng):
        def mut(win, off, m=m):
            close = off + len(m.group(0)) - 1
            return win[:close] + r"\footnote{see appendix}" + win[close:]

        edit = _window_edit(text, m.start(), mut, "fragile_cmd", rng)
        if edit:
            return edit
    return None


MUTATIONS = [typo_macro, drop_dollar, drop_brace, extra_brace, env_mismatch,
             typo_begin_env, drop_usepackage, extra_amp, drop_newcommand]
HARD_MUTATIONS = MUTATIONS + [at_distance, option_clash, fragile_cmd]


def make_break_multi(text: str, k: int, seed: int) -> tuple[str, list[dict]] | None:
    """Stack k distinct mutations. Returns (broken_text, break_edits in applied order)
    where each edit's search/replace is unique at its application step, or None."""
    rng = random.Random(f"{len(text)}:{seed}")  # mix in the doc so seeds don't pick identical classes everywhere
    fns = rng.sample(HARD_MUTATIONS, min(k, len(HARD_MUTATIONS)))
    cur, edits = text, []
    for fn in fns:
        edit = fn(cur, random.Random(rng.randrange(1 << 30)))
        if not edit:
            continue
        nxt = cur.replace(edit["search"], edit["replace"], 1)
        if nxt.count(edit["replace"]) != 1:
            continue
        cur, edits = nxt, edits + [edit]
    if len(edits) < 2:
        return None
    # fixes reverse the edits in reverse order; verify the full round trip
    doc = cur
    for e in reversed(edits):
        if doc.count(e["replace"]) != 1:
            return None
        doc = doc.replace(e["replace"], e["search"], 1)
    return (cur, edits) if doc == text else None


def make_break(text: str, mutation_name: str, seed: int) -> dict | None:
    rng = random.Random(seed)
    fn = {f.__name__: f for f in HARD_MUTATIONS}[mutation_name]
    edit = fn(text, rng)
    if not edit:
        return None
    broken = text.replace(edit["search"], edit["replace"], 1)
    # fix edit must apply unambiguously in the broken doc
    if broken.count(edit["replace"]) != 1:
        return None
    return edit
