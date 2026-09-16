"""Shared: env, edit blocks, latex log parsing (port of sundial latex-log-parser.ts),
fix prompt (port of fix-prompt.ts), local latexmk compile harness."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
CORPUS = PROJ / "corpus" / "templates"
MODEL = os.environ.get("MODEL", "thinkingmachines/Inkling-Small")
RENDERER = os.environ.get("RENDERER", "tml_v0")  # cookbook renderer name; non-Inkling bases (Qwen, gpt-oss...) use theirs


def cookbook_renderer():
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    return get_renderer(RENDERER, get_tokenizer(MODEL))


def cookbook_messages(prompt: str, target: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": prompt}]
    return msgs + ([{"role": "assistant", "content": target}] if target is not None else [])


def cookbook_text(message) -> str:
    c = message.get("content", "") if isinstance(message, dict) else message
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")

SYSTEM = """You fix LaTeX compile errors. You receive the failing file, parsed error lines, and the compile log tail. Reply ONLY with the smallest edit operations that make the document compile, keeping the content intact, in this exact format:

<edit>
<<<<<<< SEARCH
(exact text from the file)
=======
(replacement text)
>>>>>>> REPLACE
</edit>

For multi-file projects, name the file on the tag: <edit file="sections/intro.tex"> (default: the file shown).
The SEARCH text must match the file exactly and unambiguously. Do not explain."""


THINK_BUDGET = int(os.environ.get("THINK_BUDGET", "0"))
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def system_prompt() -> str:
    """Prod system prompt, plus an optional reasoning allowance (THINK_BUDGET tokens) for the thinking-budget experiments."""
    return SYSTEM + (f"\nYou may reason first, inside <think>...</think>, using at most {THINK_BUDGET} tokens; "
                     "then output only the edit blocks." if THINK_BUDGET else "")


def load_env() -> None:
    for env in (PROJ / ".env", PROJ.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    k, _, val = line.partition("=")
                    os.environ.setdefault(k.strip(), val.strip())
    if os.environ.get("TML_KEY"):
        os.environ.setdefault("TINKER_API_KEY", os.environ["TML_KEY"])


def get_renderer():
    from tml_renderers import tokenizers, v0

    return v0.Renderer(tokenizers.o200k_base_chat())


# --- edit blocks (house style, same as voice-edits) ---

EDIT_RE = re.compile(
    r"<edit>\s*<<<<<<< SEARCH\n(.*?)\n?=======\n?(.*?)>>>>>>> REPLACE\s*</edit>",
    re.DOTALL,
)


def format_edits(edits: list[tuple[str, str]]) -> str:
    blocks = []
    for search, replace in edits:
        rep = replace + "\n" if replace else ""
        blocks.append(f"<edit>\n<<<<<<< SEARCH\n{search}\n=======\n{rep}>>>>>>> REPLACE\n</edit>")
    return "\n\n".join(blocks)


def parse_edits(text: str) -> list[tuple[str, str]]:
    return [(s, r.rstrip("\n")) for s, r in EDIT_RE.findall(text)]


# --- project lane: <edit file="sections/intro.tex"> blocks; file defaults to main.tex ---

FILE_EDIT_RE = re.compile(
    r'<edit(?:\s+file="([^"]+)")?>\s*<<<<<<< SEARCH\n(.*?)\n?=======\n?(.*?)>>>>>>> REPLACE\s*</edit>',
    re.DOTALL,
)


def format_file_edits(edits: list[tuple[str, str, str]]) -> str:
    blocks = []
    for file, search, replace in edits:
        rep = replace + "\n" if replace else ""
        blocks.append(f'<edit file="{file}">\n<<<<<<< SEARCH\n{search}\n=======\n{rep}>>>>>>> REPLACE\n</edit>')
    return "\n\n".join(blocks)


def parse_file_edits(text: str) -> list[tuple[str, str, str]]:
    return [(f or "main.tex", s, r.rstrip("\n")) for f, s, r in FILE_EDIT_RE.findall(text)]


def apply_file_edits(files: dict[str, str], edits: list[tuple[str, str, str]]) -> tuple[dict[str, str], str | None]:
    """Apply blocks to a {relpath: text} project. SEARCH must match exactly once in the named file."""
    if not edits:
        return files, "no edit blocks parsed"
    files = dict(files)
    for file, search, replace in edits:
        if file not in files:
            return files, f"no such file: {file}"
        n = files[file].count(search)
        if n == 0:
            return files, f"SEARCH not found in {file}: {search[:60]!r}"
        if n > 1:
            return files, f"SEARCH ambiguous ({n} matches) in {file}: {search[:60]!r}"
        files[file] = files[file].replace(search, replace, 1)
    return files, None


TEXT_EXTS = {".tex", ".sty", ".cls", ".bib", ".bst", ".def", ".cfg", ".clo", ".txt", ".latexmkrc"}


def project_files(src_dir) -> dict[str, str]:
    """All editable text files of a project as {relpath: text} (skips build outputs)."""
    src_dir = Path(src_dir)
    out = {}
    for p in sorted(src_dir.rglob("*")):
        if p.is_file() and (p.suffix in TEXT_EXTS or p.name == ".latexmkrc") \
                and p.suffix not in (".aux", ".log", ".out", ".toc", ".bbl", ".blg"):
            out[str(p.relative_to(src_dir))] = p.read_text(errors="replace")
    return out


def tree_block(src_dir, root: str = "main.tex") -> str:
    names = sorted(str(p.relative_to(src_dir)) for p in Path(src_dir).rglob("*") if p.is_file()
                   and p.suffix not in (".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".synctex.gz")
                   and p.name != Path(root).stem + ".pdf")  # figure PDFs are listed; the root's own output is not
    shown = names[:40] + ([f"... ({len(names) - 40} more)"] if len(names) > 40 else [])
    return "\n\nProject files:\n" + "\n".join(f"- {n}" for n in shown)


CTX_MAX_FILES, CTX_FILE_CHARS, CTX_TOTAL_CHARS = 5, 6000, 14000
CTX_SKIP_CS = {"begin", "end", "documentclass", "usepackage", "input", "include", "section", "item", "label", "ref", "cite"}


def _numbered(lines: list[str], start: int = 0) -> str:
    return "\n".join(f"{start + i + 1}: {ln}" for i, ln in enumerate(lines))


def context_block(src_dir, reported: str, error_lines: list[dict], log_tail: str, root: str = "main.tex") -> str:
    """Companion files a single-shot fixer needs for multi-file projects: files in the error stack / log, the root file,
    local classes and styles, files included near the error, .bib on citation errors. Small files in full, large ones
    as their head plus lines mentioning control sequences from the log. Budgeted so the prompt stays fast."""
    files = project_files(src_dir)
    if root not in files:
        root = reported
    order: list[str] = []

    def add(name: str) -> str | None:
        n = name.strip().lstrip("./")
        for cand in (n, n + ".tex", n + ".sty", n + ".cls", n + ".bib"):
            hit = cand if cand in files else next((f for f in files if f.endswith("/" + cand)), None)
            if hit:
                if hit != reported and hit not in order:
                    order.append(hit)
                return hit
        return None

    for e in error_lines:
        if e.get("file"):
            add(e["file"])
    for m in re.finditer(r"\(\./([^\s()]+\.(?:tex|sty|cls|bib|def|cfg|clo))", log_tail):
        add(m.group(1))
    rep = files.get(reported, "")
    inc = r"\\(?:input|include|subfile|import|bibliography|addbibresource)\{([^}]+)\}"
    first = next((e["line"] for e in error_lines if isinstance(e.get("line"), int)), None)
    if first:  # files included near the reported line: the error usually surfaces at the \include
        window = "\n".join(rep.split("\n")[max(0, first - 1 - EXCERPT_RADIUS_LINES):first + EXCERPT_RADIUS_LINES])
        for m in re.finditer(inc, window):
            for name in m.group(1).split(","):
                add(name)
    if reported != root:
        add(root)
    head = files.get(root, "")[:6000] + rep
    for m in re.finditer(r"\\(?:documentclass|usepackage|RequirePackage|LoadClass)(?:\[[^\]]*\])?\{([^}]+)\}", head):
        for name in m.group(1).split(","):
            add(name)
    for m in re.finditer(inc, rep):
        for name in m.group(1).split(","):
            add(name)
    if re.search(r"cit|bib|natbib|biblatex", " ".join(e["text"] for e in error_lines), re.I):
        for f in files:
            if f.endswith(".bib"):
                add(f)
    if not order:
        return ""
    cs = {c for c in re.findall(r"\\([A-Za-z@]{3,})", log_tail[-1500:]) if c not in CTX_SKIP_CS}
    parts, total = [], 0
    at = {}  # companion -> line the log blames in it
    for e in error_lines:
        if e.get("file") and isinstance(e.get("line"), int):
            at.setdefault(add(e["file"]) or "", e["line"])
    for name in order[:CTX_MAX_FILES]:
        text = files[name]
        lines = text.split("\n")
        if len(text) <= CTX_FILE_CHARS:
            body, label = _numbered(lines), "full file"
        elif name in at:  # the log names a line in this file: show the window around it, like the reported file
            a, b = max(0, at[name] - 1 - EXCERPT_RADIUS_LINES), min(len(lines), at[name] + EXCERPT_RADIUS_LINES)
            body, label = _numbered(lines[a:b], a)[:CTX_FILE_CHARS], f"lines {a + 1}-{b}"
        else:
            keep = set([i for i, ln in enumerate(lines) if not ln.lstrip().startswith("%")][:30])  # head, comments skipped
            for i, ln in enumerate(lines):
                if any(("\\" + c) in ln for c in cs):
                    keep.update(range(max(0, i - 2), min(len(lines), i + 3)))
            body, label, prev = [], "head and lines matching the log", -1
            for i in sorted(keep):
                if i != prev + 1:
                    body.append("...")
                body.append(f"{i + 1}: {lines[i]}"); prev = i
            body = "\n".join(body)[:CTX_FILE_CHARS]
        if total + len(body) > CTX_TOTAL_CHARS:
            break
        total += len(body)
        parts.append(f"`{name}` ({label}, line-numbered):\n```\n{body}\n```")
    if not parts:
        return ""
    return "\n\nOther project files (edit them with <edit file=\"...\">):\n\n" + "\n\n".join(parts)


def apply_edits(doc: str, edits: list[tuple[str, str]]) -> tuple[str, str | None]:
    if not edits:
        return doc, "no edit blocks parsed"
    for search, replace in edits:
        n = doc.count(search)
        if n == 0:
            return doc, f"SEARCH not found: {search[:60]!r}"
        if n > 1:
            return doc, f"SEARCH ambiguous ({n} matches): {search[:60]!r}"
        doc = doc.replace(search, replace, 1)
    return doc, None


# --- log parser: faithful port of lib/workspace/latex-log-parser.ts ---

PATH_EXT_RE = re.compile(
    r"\.(?:tex|sty|cls|bib|bst|def|cfg|clo|fd|ldf|aux|toc|out|bbl)$", re.I
)
MAX_EXCERPT_LINES = 8
SOURCE_LINE_LOOKAHEAD = 40

ERROR_PREFIX_RE = re.compile(r"^! (.*)$")
SOURCE_LINE_RE = re.compile(r"^l\.(\d+)\b ?(.*)$")
FILE_LINE_RE = re.compile(r"^(.+?):(\d+):\d*:?\s*(.+)$")
WARNING_RE = re.compile(
    r"(?:^|\s)((?:LaTeX|Package [\w@-]+|LaTeX Font|pdfTeX|Class [\w@-]+) Warning:.*)$"
)
INPUT_LINE_RE = re.compile(r"on input line (\d+)")
BADBOX_RE = re.compile(r"^(Overfull|Underfull) \\([hv]box)\b(.*)$")
BADBOX_LINES_RE = re.compile(r"at lines (\d+)--\d+")
BADBOX_LINE_RE = re.compile(r"(?:at|detected at) line (\d+)")


def _normalize_path(raw: str) -> str:
    return re.sub(r"^\./", "", raw.strip())


def _looks_like_path(token: str) -> bool:
    return bool(token) and ("/" in token or PATH_EXT_RE.search(token) is not None)


def _advance_file_stack(line: str, stack: list[str]) -> None:
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "(":
            j = i + 1
            token = ""
            while j < len(line) and line[j] not in "(){} \t":
                token += line[j]
                j += 1
            if _looks_like_path(token):
                stack.append(_normalize_path(token))
                i = j - 1
        elif ch == ")":
            if stack:
                stack.pop()
        i += 1


def _file_at(stack: list[str], root_file: str | None) -> tuple[str | None, bool]:
    if stack:
        return stack[-1], True
    return root_file, False


def _clamp_excerpt(lines: list[str]) -> str:
    return "\n".join(lines[:MAX_EXCERPT_LINES]).rstrip()


def parse_latex_log(log: str, root_file: str | None = None) -> list[dict]:
    if not log or not log.strip():
        return []
    lines = log.split("\n")
    lines = [ln.rstrip("\r") for ln in lines]
    stack: list[str] = []
    items: list[dict] = []
    saw_hard_failure = False

    for i, line in enumerate(lines):
        m = ERROR_PREFIX_RE.match(line)
        if m:
            saw_hard_failure = True
            excerpt = [line]
            source_line = None
            for j in range(i + 1, min(len(lines), i + SOURCE_LINE_LOOKAHEAD)):
                excerpt.append(lines[j])
                sm = SOURCE_LINE_RE.match(lines[j])
                if sm:
                    source_line = int(sm.group(1))
                    break
            file, from_log = _file_at(stack, root_file)
            message = m.group(1).strip() or "LaTeX error"
            if message.startswith("Undefined control sequence"):  # name the macro: it sits at the end of the l.N line
                cs = re.findall(r"\\[a-zA-Z@]+|\\.", next((ln for ln in excerpt if SOURCE_LINE_RE.match(ln)), ""))
                if cs:
                    message = f"Undefined control sequence {cs[-1]}"
            items.append({
                "severity": "error", "file": file, "fileFromLog": from_log,
                "line": source_line, "message": message,
                "rawExcerpt": _clamp_excerpt(excerpt),
            })
            _advance_file_stack(line, stack)
            continue

        if BADBOX_RE.match(line):
            lm = BADBOX_LINES_RE.search(line) or BADBOX_LINE_RE.search(line)
            file, from_log = _file_at(stack, root_file)
            items.append({
                "severity": "badbox", "file": file, "fileFromLog": from_log,
                "line": int(lm.group(1)) if lm else None,
                "message": line.strip(), "rawExcerpt": line.rstrip(),
            })
            _advance_file_stack(line, stack)
            continue

        wm = WARNING_RE.search(line)
        if wm:
            excerpt = [line]
            im = INPUT_LINE_RE.search(line)
            input_line = int(im.group(1)) if im else None
            j = i + 1
            while input_line is None and j < len(lines) and j < i + MAX_EXCERPT_LINES:
                if not lines[j].strip() or ERROR_PREFIX_RE.match(lines[j]):
                    break
                excerpt.append(lines[j])
                im = INPUT_LINE_RE.search(lines[j])
                if im:
                    input_line = int(im.group(1))
                j += 1
            file, from_log = _file_at(stack, root_file)
            items.append({
                "severity": "warning", "file": file, "fileFromLog": from_log,
                "line": input_line, "message": wm.group(1).strip(),
                "rawExcerpt": _clamp_excerpt(excerpt),
            })
            _advance_file_stack(line, stack)
            continue

        fm = FILE_LINE_RE.match(line)
        if fm:
            candidate = _normalize_path(fm.group(1))
            if not re.search(r"\s", candidate) and _looks_like_path(candidate):
                saw_hard_failure = True
                items.append({
                    "severity": "error", "file": candidate, "fileFromLog": True,
                    "line": int(fm.group(2)), "message": fm.group(3).strip(),
                    "rawExcerpt": line.rstrip(),
                })
                _advance_file_stack(line, stack)
                continue

        if re.match(r"^error:", line.strip(), re.I):
            saw_hard_failure = True
        _advance_file_stack(line, stack)

    if saw_hard_failure and not any(it["severity"] == "error" for it in items):
        items.append({
            "severity": "error", "file": root_file, "fileFromLog": False, "line": None,
            "message": "Compile failed (see log)",
            "rawExcerpt": _clamp_excerpt(lines[-MAX_EXCERPT_LINES:]),
        })
    return items


def collapse_items(items: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    for it in items:
        key = f"{it['severity']}|{it['file'] or ''}|{it['line'] or ''}"
        if key in by_key:
            by_key[key]["count"] += 1
        else:
            by_key[key] = {**it, "count": 1}
    return list(by_key.values())


def error_lines_from_items(
    items: list[dict], root_file: str | None, project_files: set[str]
) -> list[dict]:
    """Port of errorLinesFromProblems: resolveLogPath = 'in project files or None'."""
    out = []
    for it in collapse_items(items):
        if it["severity"] != "error" or not isinstance(it["line"], int) or it["line"] < 1:
            continue
        path = it["file"] if it["file"] in project_files else None
        out.append({
            "line": it["line"],
            "text": it["message"][:160],
            "file": path,
            "fileLabel": (None if path == root_file else os.path.basename(path)) if path
            else (os.path.basename(it["file"]) if it["file"] else None),
        })
    return out[:12]


# --- fix prompt: port of lib/latex/fix-prompt.ts ---

MAX_LOG_CHARS = 1500
MAX_ERROR_LINES = 12
MAX_FULL_SOURCE_LINES = 120
EXCERPT_RADIUS_LINES = 25
MAX_SOURCE_CHARS = 4000


def _source_block(tex_path: str, source_text: str, error_lines: list[dict],
                  full_lines: int = MAX_FULL_SOURCE_LINES,
                  max_chars: int = MAX_SOURCE_CHARS) -> str:
    lines = source_text.split("\n")
    start, end = 0, len(lines)
    if len(lines) > full_lines:
        first = next((e["line"] for e in error_lines if not e["fileLabel"]), None)
        if first is None:
            first = error_lines[0]["line"] if error_lines else None
        if not isinstance(first, int):
            return ""
        start = max(0, first - 1 - EXCERPT_RADIUS_LINES)
        end = min(len(lines), first + EXCERPT_RADIUS_LINES)
    excerpt = "\n".join(f"{start + i + 1}: {ln}" for i, ln in enumerate(lines[start:end]))
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
    if not excerpt.strip():
        return ""
    label = "full file" if start == 0 and end == len(lines) else f"lines {start + 1}-{end}"
    return f"\n\nCurrent content of `{tex_path}` ({label}, line-numbered):\n```\n{excerpt}\n```"


def build_fix_prompt(
    tex_path: str, error_lines: list[dict], log_text: str, source_text: str | None,
    full_lines: int = MAX_FULL_SOURCE_LINES, max_chars: int = MAX_SOURCE_CHARS,
    tree: str = "",
) -> str:
    # Header matches prod fix-prompt.ts minus the agent-loop instructions (shell/tool
    # sentences) — this path is single-shot: the reply is edit blocks, nothing else.
    header = (
        f"The LaTeX file `{tex_path}` fails to compile. Apply the smallest edit that fixes the "
        "compile errors and keep my content intact."
    )
    n_src = source_text.count("\n") + 1 if isinstance(source_text, str) else None
    lines = []
    for e in error_lines[:MAX_ERROR_LINES]:
        if not e["fileLabel"] and n_src and isinstance(e["line"], int) and e["line"] > n_src:  # a package-internal line misattributed to the root file
            lines.append(f"- inside a package (its line {e['line']}, not `{tex_path}`): {e['text']}")
        else:
            lines.append(f"- {(e['file'] or e['fileLabel']) + ' ' if e['fileLabel'] else ''}line {e['line']}: {e['text']}")
    error_block = "\nErrors:\n" + "\n".join(lines) if lines else ""
    source_block = (
        _source_block(tex_path, source_text, error_lines, full_lines, max_chars)
        if isinstance(source_text, str) and source_text.strip()
        else ""
    )
    tail = log_text[-MAX_LOG_CHARS:] if len(log_text) > MAX_LOG_CHARS else log_text
    log_block = f"\n\nCompile log (tail):\n```\n{tail.strip()}\n```" if tail.strip() else ""
    return f"{header}{error_block}{tree}{source_block}{log_block}"


# --- compile harness ---

def pick_engine(text: str, src_dir: Path | None = None) -> str:
    """Engine from the root file's preamble; local .cls/.sty files count too (awesome-cv loads fontspec from the class)."""
    if src_dir is not None:
        text += "".join(p.read_text(errors="replace")[:20000] for p in Path(src_dir).glob("*") if p.suffix in (".cls", ".sty"))
    if re.search(r"\\(usepackage|RequirePackage)(\[[^\]]*\])?\{[^}]*(fontspec|polyglossia|unicode-math|xeCJK)", text):
        return "xelatex"
    if re.search(r"\\(usepackage|RequirePackage)(\[[^\]]*\])?\{[^}]*(luacode|luatextra)|\\directlua", text):
        return "lualatex"
    return "pdflatex"


ENGINE_FLAG = {"pdflatex": "-pdf", "xelatex": "-pdfxe", "lualatex": "-pdflua"}


BLG_FATAL_RE = re.compile(r"^(I couldn't open (?:style|database) file .*|I found no \\\\\w+ commands?.*|"
                          r"ERROR - .*|I couldn't find .*|I was expecting .*)$", re.M)


def texlive_version(log: str) -> str | None:
    """'TeX Live 2026' from the engine banner on the log's first line (the farm image is texlive/texlive:latest)."""
    m = re.search(r"TeX Live (\d{4})", log[:300])
    return m.group(1) if m else None


def compile_project(
    src_dir: Path, bundles: list[Path], root: str = "main.tex",
    overrides: dict[str, str] | None = None, timeout: int = 120, bib: bool = False, backend: str | None = None,
) -> dict:
    """Copy src_dir to a temp dir (applying content overrides), run latexmk.
    Returns {ok, log, items, error_items, seconds}."""
    import time

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copytree(src_dir, d, dirs_exist_ok=True)
        for stale in Path(d).glob(Path(root).stem + ".*"):  # never let a shipped PDF/log stand in for this compile
            if stale.suffix in (".pdf", ".log", ".aux"):
                stale.unlink()
        for rel, content in (overrides or {}).items():
            (d / rel).write_text(content)
        root_text = (d / root).read_text(errors="replace")
        engine = pick_engine(root_text, d)
        t0 = time.monotonic()
        if (backend or os.environ.get("COMPILE_BACKEND")) == "modal" and not bundles:  # remote farm (modal/modal_compile.py)
            r = _remote_compile(d, root, ENGINE_FLAG[engine], bib, timeout)
            items = parse_latex_log(r["log"], root_file=root)
            for m in (BLG_FATAL_RE.finditer(r["blg"]) if bib else ()):
                items.append({"severity": "error", "file": None, "fileFromLog": True, "line": None,
                              "message": "bibtex: " + m.group(1).strip(), "rawExcerpt": m.group(1)})
            error_items = [it for it in items if it["severity"] == "error"]
            ok = r["pdf_exists"] and not error_items and not r["timed_out"]
            return {"ok": ok, "log": r["log"], "items": items, "error_items": error_items, "engine": engine,
                    "seconds": round(time.monotonic() - t0, 2), "timed_out": r["timed_out"], "texlive": texlive_version(r["log"]),
                    "pdf_text": r["pdf_text"] if ok else None, "pdf_pages": r["pdf_pages"] if ok else None,
                    "pdf_img": r.get("pdf_img") if ok else None}
        texinputs = ".:" + ":".join(str(b) for b in bundles) + ":"
        env = {**os.environ, "TEXINPUTS": texinputs, "BSTINPUTS": texinputs}
        proc = subprocess.Popen(
            ["latexmk", ENGINE_FLAG[engine], "-interaction=nonstopmode"]
            + ([] if bib else ["-bibtex-"]) + [root],
            cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True,  # own process group: kill pdflatex too
        )
        try:
            proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, 9)  # latexmk AND its pdflatex children (no more orphans)
            except ProcessLookupError:
                pass
            proc.wait()
        seconds = time.monotonic() - t0
        log_path = d / (Path(root).stem + ".log")
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        pdf = d / (Path(root).stem + ".pdf")
        items = parse_latex_log(log, root_file=root)
        if bib:  # bibtex/biber failures never appear as `!` lines; surface them from the .blg
            blg = d / (Path(root).stem + ".blg")
            for m in BLG_FATAL_RE.finditer(blg.read_text(errors="replace") if blg.exists() else ""):
                items.append({"severity": "error", "file": next((n for n in os.listdir(d) if n.endswith(".bib")), None),
                              "fileFromLog": True, "line": None, "message": "bibtex: " + m.group(1).strip(),
                              "rawExcerpt": m.group(1)})
        error_items = [it for it in items if it["severity"] == "error"]
        ok = pdf.exists() and not error_items and not timed_out
        pdf_text, pdf_pages = None, None
        if ok:
            pdf_text, pdf_pages = pdf_metrics(pdf)
        return {"ok": ok, "log": log, "items": items, "error_items": error_items,
                "seconds": round(seconds, 2), "engine": engine, "timed_out": timed_out, "texlive": texlive_version(log),
                "pdf_text": pdf_text, "pdf_pages": pdf_pages}


_remote_fn = None
_spawn_pool = None


def _remote_compile(d: Path, root: str, engine_flag: str, bib: bool, timeout: int) -> dict:
    global _remote_fn
    import io
    import tarfile

    import modal
    if _remote_fn is None:
        _remote_fn = modal.Function.from_name(os.environ.get("MODAL_APP", "latexfix-compile4"), "compile_tar")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(d, arcname=".")
    global _spawn_pool
    from concurrent.futures import ThreadPoolExecutor
    if _spawn_pool is None:
        _spawn_pool = ThreadPoolExecutor(64)
    call = None
    try:
        # spawn() can hang forever on a broken Modal channel after a network blip: run it with its own deadline
        call = _spawn_pool.submit(_remote_fn.spawn, buf.getvalue(), root, engine_flag, bib, timeout).result(timeout=120)
        return call.get(timeout=timeout + 90)  # a hung call must not stall a whole RL batch
    except Exception as e:  # noqa: BLE001  (TimeoutError or transport error): scored as a failed compile, flagged timed_out
        if call is not None:
            try:
                call.cancel()
            except Exception:  # noqa: BLE001
                pass
        print(f"[compile farm] {type(e).__name__} for {root}: scored as timed out", file=sys.stderr)
        return {"log": "", "blg": "", "timed_out": True, "pdf_exists": False, "pdf_text": None, "pdf_pages": None, "pdf_img": None}


def pdf_metrics(pdf: Path) -> tuple[str | None, int | None]:
    """Whitespace-normalized extracted text + page count (ground-truth visual proxy)."""
    try:
        text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                              timeout=30).stdout.decode(errors="replace")
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, timeout=30
                              ).stdout.decode(errors="replace")
        pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        return " ".join(text.split()), pages
    except Exception:  # noqa: BLE001
        return None, None


def pdf_similarity(a: str | None, b: str | None, ascii_only: bool = False) -> float | None:
    """ascii_only: drop non-ASCII glyphs first -- pdftotext maps math glyphs differently across TeX Live releases
    (no ToUnicode cmaps before ~2022), which is noise when the reference was rendered by an era image."""
    if not a or not b:
        return None
    import difflib

    if ascii_only:
        a, b = (re.sub(r"[^\x20-\x7e\s]+", "", s) for s in (a, b))
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


IMG_W, IMG_H = 48, 64  # modal_compile._page_images raster size


def _blocks(img: bytes, f: int) -> list[float]:
    """Mean ink (0..255, 255 = black) per f x f block: coarse so a one-line reflow does not zero the score."""
    w, h = IMG_W // f, IMG_H // f
    out = []
    for by in range(h):
        for bx in range(w):
            tot = 0
            for y in range(by * f, by * f + f):
                row = img[y * IMG_W + bx * f: y * IMG_W + bx * f + f]
                tot += f * 255 - sum(row)
            out.append(tot / (f * f))
    return out


def image_similarity(a: list[str] | None, b: list[str] | None, factor: int = 4) -> float | None:
    """Page-raster agreement in [0, 1] (1 = identical ink layout): mean over pages of 1 - |A-B| / (ink(A) + ink(B)) on
    factor x factor block densities; a page one side lacks scores 0. Sees figures, TikZ and layout that pdftotext cannot."""
    if not a or not b:
        return None
    import base64
    n = max(len(a), len(b))
    sims = []
    for i in range(n):
        if i >= len(a) or i >= len(b):
            sims.append(0.0)
            continue
        x, y = (_blocks(base64.b64decode(s), factor) for s in (a[i], b[i]))
        ink = sum(x) + sum(y)
        sims.append(1.0 if ink == 0 else max(0.0, 1.0 - sum(abs(u - v) for u, v in zip(x, y)) / ink))
    return sum(sims) / n


def text_similarity(a: str | None, b: str | None, ascii_only: bool = False) -> float | None:
    """Continuous text agreement: character-level on short documents (a 13-word title page must not lose 8 points per
    word), word-level beyond 3,000 chars (SequenceMatcher is quadratic)."""
    if not a or not b:
        return None
    import difflib

    if ascii_only:
        a, b = (re.sub(r"[^\x20-\x7e\s]+", "", s) for s in (a, b))
    a, b = " ".join(a.split()), " ".join(b.split())
    if max(len(a), len(b)) <= 3000:
        return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def pdf_score(res: dict, ref: dict) -> dict:
    """All similarity views of a compiled result against a reference: word (strict's metric), text (continuous), image."""
    asc = ref.get("ascii", False)
    out = {"pdf_sim": pdf_similarity(res["pdf_text"], ref.get("text"), asc),
           "text_sim": text_similarity(res["pdf_text"], ref.get("text"), asc),
           "img_sim": image_similarity(res.get("pdf_img"), ref.get("img")),
           "pages_ok": res["pdf_pages"] == ref.get("pages")}
    out["strict"] = bool(out["pages_ok"] and (out["pdf_sim"] or 0) >= ref.get("sim_min", 0.985))
    sims = [v for v in (out["text_sim"], out["img_sim"]) if v is not None]
    out["sim"] = sum(sims) / len(sims) if sims else None
    return out


REFS_DEFAULT = "refs.json,refs_pool.json,refs_skew.json"  # human refs, teacher pseudo-refs, historic-TeX-Live refs


def load_refs() -> dict:
    """Reference PDFs keyed by doc; REFS=<comma list> picks the set (v2: refs2.json,refs2_skew.json,refs2_pool.json)."""
    refs = {}
    for name in os.environ.get("REFS", REFS_DEFAULT).split(","):
        if (DATA / name.strip()).exists():
            refs.update(json.loads((DATA / name.strip()).read_text()))
    return refs

