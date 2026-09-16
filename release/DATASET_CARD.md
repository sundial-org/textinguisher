# TeXtinguisher: a benchmark of real LaTeX compile errors

Real LaTeX projects that fail to compile with a located error, paired (where a human fix exists) with the
smallest SEARCH/REPLACE edit that makes them compile. Verifier: `latexmk` (TeX Live 2025/2026). Built by
`scripts/build_release.py`; integrity in `MANIFEST.json` (rows, bytes, sha256 per file).

## Files

| file | rows | what |
|---|--:|---|
| `texse_heldout.jsonl` | 287 | TeX.SE single-file benchmark (question's MWE + accepted answer's document) |
| `texse_train.jsonl` | 922 | training split of the same mining run |
| `texse2_heldout.jsonl` | 647 | TeX.SE relaxed mining (error mentioned in title/body; fix from accepted or top answer); 22 rows dropped after an audit found their reference broken |
| `texse2_train.jsonl` | 2122 | its training split |
| `texse_pool.jsonl` / `texse2_pool.jsonl` | 1774 / 3923 | unlabeled failing MWEs (no human fix; compile-reward only) |
| `multi2_heldout.jsonl` / `multi2_train.jsonl` | 30 / 87 | multi-file projects: TeX.SE questions with companion files + GitHub commits |
| `skew_eval.jsonl`, `skew_multi_heldout.jsonl`, `skew_train.jsonl` | 184 / 200 / 1628 | arXiv "toolchain skew" rows, metadata only (see below) |
| `refs.json` | 7424 docs | PDF text + page count of the fixed document, keyed by `doc` (`pseudo: true` = teacher-generated, pool rows only) |
| `refs_verified.json` | 433 docs | held-out references whose human fix was confirmed by an independent teacher model or a judge (`how`), recompiled on TeX Live 2026; use these for content-preserving metrics |
| `projects/<id>/` | 9814 dirs | sources needed to reproduce the compile (one per unique id) |

Row ids are unique; `id` names the project directory, `doc` the reference entry.

## Row format

Common fields: `id`, `doc`, `engine` (pdflatex/xelatex/lualatex), `error_lines` (parsed `{file, line, text}`),
`license`; non-skew rows add `log_tail`, `prompt` (the production fix prompt: error lines + source window + log tail),
`error_category`. Skew rows are metadata only (no project directory, no `prompt`/`log_tail`/`error_category`):
`arxiv_id`, `year`, `hist_year`, `signature`, `first_error`, `missing_file`, `n_files`.

- Single-file rows: `projects/<id>/main.tex` is the **fixed** document; apply `break_edits` (replace `search` by
  `replace`, once, in order) to obtain the failing document. `target` is the fix as `<edit>` SEARCH/REPLACE blocks.
  Pool rows have no `break_edits`/`target`; their `main.tex` is the failing document.
- Multi-file rows (`lane: project`): `projects/<id>/` is the fixed project, `file_edits` break it, `target` uses
  `<edit file="...">` blocks. `prompt` is the project-context prompt (file tree + reported file); `prompt_v1` is the
  single-file prompt for the reported file only. `reported_file`, `fix_files`, `changed_files`, `n_files` describe the row.
- TeX.SE provenance: `question_id`, `question_url`, `title`, `tags`, `score`, `created`, `question_author`
  (`{user_id, name, url}`), `license`; `answer_id`/`accepted_answer_id`, `answer_url`, `answer_author`, `answer_license`
  only when an answer was used (absent on 3641 pool rows).
- GitHub provenance: `repo`, `repo_url`, `sha` (fixing commit), `parent_sha` (failing state), `url` (commit), `license`.

Local absolute paths were rewritten: the project path to `projects/<id>`, other `/Users/<name>/` prefixes to `~/`.
Absolute paths that remain are content of the posts themselves (e.g. `\graphicspath{{/home/...}}`).

## How rows were mined and verified

- **TeX.SE** (`scripts/mine_texse.py`, `mine_texse2.py`, `mine_texse_pool.py`, `mine_texse_multi.py`): from the
  archive.org data dump `stackexchange_20251231`. A question qualifies if its body has a full `\documentclass ...
  \end{document}` MWE and reports an error (`texse`: a pasted `! ...` line; `texse2`: an error mention in title/body).
  The MWE must fail locally with an error located in `main.tex`; the fix document (longest complete document in the
  accepted answer, else the top answer with score >= 2 for `texse2`; otherwise a snippet) must compile clean. Diff
  capped at 6 hunks / 40 lines; edits must apply uniquely and round-trip exactly. Rows whose first error is a missing
  file/font/graphic or an engine mismatch (fontspec, Unicode engine) were dropped, so those classes are absent.
  Funnels and error histograms: `data/texse_report.md`, `texse2_report.md`, `texse_pool_report.md`, `multi2_report.md`.
- **GitHub multi-file** (`scripts/mine_github_multi.py`, `mine_github_multi2.py`): commit pairs where the parent
  fails with a located error and the child compiles, edits round-trip exactly, max 5 rows per repo. 13 `github_multi`
  rows were never re-verified locally and have no reference entry; they are not in this release (unlicensed anyway).
- **Splits**: by question id (stratified by error category) for TeX.SE, by repo/question for multi-file, by arXiv id
  for skew. Held-out ids, question ids, repos and arXiv ids do not appear in any train or pool file (checked by
  `build_release.py --check`, which also reports near-duplicates: Jaccard >= 0.9 on normalized `main.tex` lines finds
  one, `texse2_437213` vs pool row `texse2_pool_437217`, a follow-up question). Note that `texse_heldout`
  over-represents rows that were probed before the split (not a random sample).

## Metrics (`scripts/run_eval.py`)

1. **compiles** (`fixed`): the model's edits apply and `latexmk` exits clean. Headline metric; the only one for pool
   and skew rows.
2. **content kept** (`del<=40` in `scripts/heldout_table.py`): compiles and the edits remove at most 40 characters of
   non-comment content (`fix_env.deleted_chars`). A heuristic guard against "delete until it compiles"; it does not see
   content hidden by `\iffalse` or comment environments.
3. **strict**: compiles and the PDF text matches the reference (`refs.json`): same page count and word-level
   `SequenceMatcher` ratio >= 0.985. Penalizes valid alternative fixes.
   On the verified subset (`refs_verified.json`, `REFS=refs2_heldout.json scripts/heldout_table.py <set>`) strict is a fair "exact fix"
   measure; on the rest, TeX.SE answers often rewrite the document, so strict under-counts correct minimal fixes.
4. **exact**: the patched source equals the human-fixed source. TeX.SE labels are whole answer documents, often
   restyled (28% of `texse` rows have >= 20 diff lines), so `exact`/`strict` under-count correct minimal fixes;
   filter on `diff_lines` for a minimal-fix subset.

## Licences and attribution

| source | licence | rows | attribution required |
|---|---|--:|---|
| TeX.SE (`texse*`, multi `source: texse`) | CC BY-SA 2.5 / 3.0 / 4.0 per post (`ContentLicense` in the dump) | 49 / 5014 / 4717 | author name + profile link (`question_author`, `answer_author`) and post links (`question_url`, `answer_url`) |
| GitHub (`multi2*`, `source: github`) | MIT 17, GPL-3.0 4, Apache-2.0 4, CC0-1.0 4, AGPL-3.0 2, CC-BY-SA-4.0 2, CC-BY-4.0 1 | 34 | per the repo's LICENSE; `repo_url`, `sha`, `license` per row |
| arXiv (`skew*`) | mostly arXiv non-exclusive-distrib 1.0 (1872 / 2012), some CC BY / BY-SA / BY-NC | 2012 | metadata only shipped |

TeX.SE `license` is the question's; on 125 rows `answer_license` differs (78 newer, 47 older; each post carries its own
`ContentLicense`). Fixed documents are derived from both posts: the TeX.SE portion is share-alike.

## Excluded and why

- `github_forks` rows (22; 7 were in `multi2_heldout`): students' theses / CVs on templates, no licence on the fork
  author's text. Also removes the 3 CJK rows that are unfixable under the eval harness's engine choice.
- GitHub rows with `license` None or NOASSERTION (97 of 131 non-fork GitHub rows).
- arXiv sources (`skew*`, `multi_synth`, `project_*`, `eval`, `hard_eval*`): arXiv's non-exclusive licence does not
  permit redistribution; skew rows keep the metadata fields listed under Row format.
- `project_train_all.jsonl`, `project_eval.jsonl` (Sundial user projects), distilled/teacher files, and everything under
  `corpus/` not referenced above.
- `refs.json` for skew rows does not exist (no clean baseline compile by construction).

## Regenerating skew rows

`scripts/gen_skew.py --target 200 --seed 3` (arXiv 2008-2018 single-root papers compiled under TeX Live 2025) and
`scripts/gen_skew_train.py` (2010-2020 multi-file papers, historical TeX Live) download the sources from arXiv and
rebuild `corpus/arxiv_skew*/<arxiv_id>/files`; match rows by `arxiv_id`. Every skew row is an organic failure (no
planted error), so there is no ground-truth patch and only `compiles` applies.

## Contact

Sundial (see repository)
