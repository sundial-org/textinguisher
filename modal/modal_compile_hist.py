"""Historic TeX Live farm: one latexmk function per year on texlive/texlive:TL<year>-historic, used to render the
era-matched reference PDF for organic arXiv skew rows (scripts/gen_skew_train.py).
Deploy: MODAL_PROFILE=sundial-prod modal deploy modal/modal_compile_hist.py
Use:    modal.Function.from_name("latexfix-compile-hist", f"compile_tl{year}") with modal_compile.compile_tar's signature.
"""
import modal

from modal_compile import _compile

YEARS = range(2013, 2021)
app = modal.App("latexfix-compile-hist")
def _make(year: int):
    def fn(tar_bytes: bytes, root: str, engine_flag: str, bib: bool, timeout: int) -> dict:
        return _compile(tar_bytes, root, engine_flag, bib, timeout)
    fn.__name__ = fn.__qualname__ = f"compile_tl{year}"  # Modal resolves functions by module-level name
    image = modal.Image.from_registry(f"texlive/texlive:TL{year}-historic", add_python="3.12").apt_install("poppler-utils").pip_install("pillow").add_local_python_source("modal_compile")
    return app.function(image=image, cpu=1.0, memory=1024, timeout=300, max_containers=40)(fn)


for year in YEARS:
    globals()[f"compile_tl{year}"] = _make(year)
