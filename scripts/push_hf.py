"""Upload release/ to a private Hugging Face dataset repo.

usage: push_hf.py [repo_id]   (default sundial/textinguisher; needs `hf auth login` first)
The 11,672 project files go up as one projects.tar.gz; everything else as-is. DATASET_CARD.md becomes README.md
with the YAML header Hugging Face expects.
"""
import os, subprocess, sys, tempfile
from huggingface_hub import HfApi

REPO = sys.argv[1] if len(sys.argv) > 1 else "sundial/textinguisher"
REL = os.path.join(os.path.dirname(__file__), "..", "release")
HEADER = """---
license: cc-by-sa-4.0
language: [en]
pretty_name: TeXtinguisher
task_categories: [text-generation]
tags: [latex, program-repair, benchmark, compile-errors]
size_categories: [10K<n<100K]
---
"""

api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    open(os.path.join(tmp, "README.md"), "w").write(HEADER + open(os.path.join(REL, "DATASET_CARD.md")).read())
    subprocess.run(["tar", "-czf", os.path.join(tmp, "projects.tar.gz"), "-C", REL, "projects"], check=True)
    for f in sorted(os.listdir(REL)):
        if f != "projects" and f != "DATASET_CARD.md":
            os.symlink(os.path.abspath(os.path.join(REL, f)), os.path.join(tmp, f))
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=tmp, commit_message="TeXtinguisher release")
print("uploaded to https://huggingface.co/datasets/" + REPO + " (private)")
