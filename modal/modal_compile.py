"""Remote latexmk farm on Modal (image texlive/texlive:latest, TeX Live 2026 at the final evaluations; results record the
version) so RL rollouts and mining are not bound by local cores.
Deploy: MODAL_PROFILE=sundial-prod modal deploy modal/modal_compile.py
Use:    COMPILE_BACKEND=modal (common.compile_project ships the project dir as a tar and parses the log locally).
"""
import io
import os
import subprocess
import tarfile
import tempfile

import modal

app = modal.App(os.environ.get("MODAL_APP", "latexfix-compile4"))  # 4 compiles per container: the workspace cap counts containers, not CPUs
image = modal.Image.from_registry("texlive/texlive:latest", add_python="3.12").apt_install("poppler-utils").pip_install("pillow")


def _compile(tar_bytes: bytes, root: str, engine_flag: str, bib: bool, timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
            tf.extractall(d, filter="data")
        cmd = ["latexmk", engine_flag, "-interaction=nonstopmode"] + ([] if bib else ["-bibtex-"]) + [root]
        timed_out = False
        try:
            subprocess.run(cmd, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        stem = os.path.splitext(root)[0]
        read = lambda n: open(os.path.join(d, n), errors="replace").read() if os.path.exists(os.path.join(d, n)) else ""
        out = {"log": read(stem + ".log"), "blg": read(stem + ".blg"), "timed_out": timed_out,
               "pdf_exists": os.path.exists(os.path.join(d, stem + ".pdf")), "pdf_text": None, "pdf_pages": None, "pdf_img": None}
        if out["pdf_exists"]:
            pdf = os.path.join(d, stem + ".pdf")
            out["pdf_text"] = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True, timeout=60).stdout
            info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True, timeout=30).stdout
            out["pdf_pages"] = next((int(l.split()[-1]) for l in info.splitlines() if l.startswith("Pages:")), None)
            out["pdf_img"] = _page_images(pdf, d)
        return out


IMG_W, IMG_H, IMG_PAGES = 48, 64, 16


def _page_images(pdf: str, d: str) -> list[str] | None:
    """Low-res grayscale rasters of the first IMG_PAGES pages (base64 of IMG_W x IMG_H bytes each): a picture-aware
    similarity signal for the reward, since pdftotext is blind to figures and TikZ."""
    import base64
    import glob
    try:
        from PIL import Image
        subprocess.run(["pdftoppm", "-r", "20", "-gray", "-png", "-l", str(IMG_PAGES), pdf, os.path.join(d, "pg")],
                       capture_output=True, timeout=120)
        imgs = []
        for f in sorted(glob.glob(os.path.join(d, "pg-*.png"))):
            im = Image.open(f).convert("L").resize((IMG_W, IMG_H), Image.BOX)
            imgs.append(base64.b64encode(im.tobytes()).decode())
        return imgs
    except Exception:  # noqa: BLE001
        return None


@app.function(image=image, cpu=4.0, memory=4096, timeout=300, max_containers=100)
@modal.concurrent(max_inputs=4)
def compile_tar(tar_bytes: bytes, root: str, engine_flag: str, bib: bool, timeout: int) -> dict:
    return _compile(tar_bytes, root, engine_flag, bib, timeout)
