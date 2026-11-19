"""Minimal Markdown → HTML for the guides in docs/ (headings, paragraphs, lists, tables, bold, code) – no dependency."""
from __future__ import annotations

import html
import re


def _inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


def render(md: str, title: str = "PV-Mapper") -> str:
    out, para, lst, tbl = [], [], None, []

    def flush():
        nonlocal para, lst, tbl
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para = []
        if lst:
            out.append(f"<{lst[0]}>" + "".join(f"<li>{_inline(x)}</li>" for x in lst[1]) + f"</{lst[0]}>")
            lst = None
        if tbl:
            rows = [r for r in tbl if not re.match(r"^\|[\s:|-]+\|$", r)]
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            h = "".join(f"<th>{_inline(c)}</th>" for c in cells[0])
            b = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in cells[1:])
            out.append(f"<table><tr>{h}</tr>{b}</table>")
            tbl = []

    for line in md.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush()
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            flush()
            n = len(m.group(1))
            out.append(f"<h{n}>{_inline(m.group(2))}</h{n}>")
            continue
        if s.startswith("|"):
            if para or lst:
                flush()
            tbl.append(s)
            continue
        m = re.match(r"^\s*(?:([-*])|(\d+)\.)\s+(.*)", s)
        if m and not line.startswith("    "):
            kind = "ol" if m.group(2) else "ul"
            if para or tbl or (lst and lst[0] != kind):
                flush()
            lst = lst or (kind, [])
            lst[1].append(m.group(3))
            continue
        if lst and line.startswith(" "):
            lst[1][-1] += " " + s.strip()
            continue
        if tbl:
            flush()
        para.append(s.strip())
    flush()
    css = ("body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#1b2430}"
           "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccd;padding:4px 8px;vertical-align:top;text-align:left}"
           "th{background:#eef1f6}code{background:#eef1f6;padding:0 3px}")
    return f"<!doctype html><html lang=de><meta charset=utf-8><title>{html.escape(title)}</title><style>{css}</style><body>{''.join(out)}</body></html>"
