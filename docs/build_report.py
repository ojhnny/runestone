"""Render docs/design_report.md to docs/design_report.pdf.

Markdown -> HTML (python-markdown) -> PDF (headless Chrome's print engine).
Figures are pulled from outputs/figures, so re-run after `make pipeline`.

    uv run python docs/build_report.py
    CHROME=/path/to/chrome uv run python docs/build_report.py   # if Chrome isn't found
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent

CANDIDATES = [
    os.environ.get("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]

CSS = """
@page { size: Letter; margin: 0.8in 0.75in; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 22pt; margin: 0 0 4pt; color: #0b0b0b; }
h2 { font-size: 15pt; margin-top: 22pt; padding-bottom: 3pt; border-bottom: 1.5px solid #2a78d6; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 14pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin-top: 10pt; page-break-after: avoid; }
h2.newpage { page-break-before: always; }
p, li { orphans: 3; widows: 3; }
code { font-family: Menlo, Consolas, monospace; font-size: 9pt; background: #f2f1ee; padding: 0 3px; border-radius: 3px; }
pre { background: #f6f5f2; border: 1px solid #e3e1db; padding: 8px 10px; border-radius: 4px; font-size: 8.6pt; line-height: 1.35; overflow-x: hidden; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; page-break-inside: avoid; }
th { background: #eef3fb; text-align: left; }
th, td { border: 1px solid #d8d6cf; padding: 4px 6px; vertical-align: top; }
img { max-width: 100%; max-height: 4.3in; object-fit: contain; display: block; margin: 8pt auto 2pt; page-break-inside: avoid; }
.caption, em.caption { display: block; text-align: center; color: #52514e; font-size: 9pt; margin-bottom: 10pt; }
blockquote { border-left: 3px solid #2a78d6; margin: 8pt 0; padding: 2pt 10pt; color: #333; background: #f7f9fd; }
.toc ul { list-style: none; padding-left: 14pt; }
.toc > ul { padding-left: 0; }
.subtitle { color: #52514e; font-size: 11pt; margin-bottom: 14pt; }
"""


def find_chrome() -> str:
    for path in CANDIDATES:
        if path and Path(path).exists():
            return path
    raise SystemExit("Chrome/Chromium not found. Set CHROME=/path/to/chrome.")


def main() -> None:
    text = (DOCS / "design_report.md").read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html"])
    body = body.replace('src="../outputs/', f'src="{(ROOT / "outputs").as_uri()}/')
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>SO-ELL design report</title><style>{CSS}</style></head><body>{body}</body></html>"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        out = DOCS / "design_report.pdf"
        subprocess.run(
            [find_chrome(), "--headless", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
             "--no-pdf-header-footer", f"--print-to-pdf={out}", page.as_uri()],
            check=True,
            capture_output=True,
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
