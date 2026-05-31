"""Sample movie graph used by the demo notebooks.

`sample_kb.txt` and `sample_qa.txt` are a small, hand-authored set of public
movie facts written in the MetaQA triple format (`subject|relation|object`).
They are committed in this directory, so every demo runs offline with no
download. This sample is NOT derived from the MetaQA dataset files; only the
triple format and relation names follow MetaQA's schema
(Zhang et al., 2018, https://github.com/yuyuz/MetaQA). The full MetaQA dataset
(CC BY 3.0) is not redistributed here; see README.md to obtain it.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent
KB_FILE = DATA_DIR / "sample_kb.txt"
QA_FILE = DATA_DIR / "sample_qa.txt"
QA_2HOP_FILE = DATA_DIR / "sample_qa_2hop.txt"
SAMPLE_FILE = DATA_DIR / "metaqa_sample.gml"


def ensure_data() -> None:
    """Build metaqa_sample.gml from the committed sample KB. Offline, idempotent.

    Notebooks call this at startup so first-time users do not have to invoke
    build_sample manually. The build only runs when sample.gml is absent.
    """
    from . import build_sample

    if not SAMPLE_FILE.exists():
        build_sample.build_sample(KB_FILE, [QA_FILE], SAMPLE_FILE)
