#!/usr/bin/env python3
"""Generate the static GitHub Pages docs site from docs/content/*.md.

Imports backend/docs.py so the site and the in-app Help tab share ONE renderer.
Pure stdlib. Output defaults to site/. Run: python tools/build_docs.py
"""
import html, os, re, shutil, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend"))
import docs  # noqa: E402

# The shared renderer emits sibling crosslinks as `slug.md` (the canonical
# reference the in-app viewer resolves by slug). On the static site those must
# point at the generated `slug.html`. Rewrite only relative .md hrefs, keeping
# any #anchor - absolute links (http(s)/mailto), root-relative, and image
# paths are left untouched.
_MD_LINK = re.compile(r'href="(?!https?://|mailto:|/|#)([^":#]+)\.md(#[^"]*)?"')


def _md_links_to_html(body):
    return _MD_LINK.sub(lambda m: 'href="%s.html%s"' % (m.group(1), m.group(2) or ""), body)

_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — LlamaForge Docs</title>
<style>
:root{{--bg:#080a0b;--panel:#0f1315;--hair:#1e262a;--ink:#c8d2d4;--dim:#6b7a7e;
--amber:#ffb000;--cyan:#3fd7e6;--green:#39d98a;--red:#ff5c57;--code-bg:#060809;--code-ink:#9fb0b2;--ink-strong:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.6 ui-monospace,Menlo,Consolas,monospace;display:grid;grid-template-columns:240px 1fr}}
nav.side{{background:var(--panel);border-right:1px solid var(--hair);padding:18px;height:100vh;overflow:auto;position:sticky;top:0}}
nav.side .brand{{color:var(--amber);font-weight:700;letter-spacing:.1em;margin-bottom:14px}}
nav.side .sec{{color:var(--ink);text-transform:uppercase;font-size:11px;letter-spacing:.08em;margin:12px 0 4px}}
nav.side a{{display:block;color:var(--dim);text-decoration:none;padding:2px 0;font-size:12px}}
nav.side a.active,nav.side a:hover{{color:var(--amber)}}
main{{padding:28px 40px;max-width:820px}}
h1,h2,h3{{color:var(--ink-strong)}}
pre{{background:var(--code-bg);color:var(--code-ink);padding:12px;overflow:auto;border:1px solid var(--hair)}}
code{{background:var(--code-bg);color:var(--code-ink);padding:1px 4px}}pre code{{padding:0}}
table{{border-collapse:collapse}}th,td{{border:1px solid var(--hair);padding:4px 8px}}
img{{max-width:100%;border:1px solid var(--hair)}}a{{color:var(--cyan)}}
.admon{{border-left:3px solid var(--amber);background:var(--panel);padding:8px 12px;margin:10px 0}}
.admon-warning{{border-left-color:var(--red)}}.admon-tip{{border-left-color:var(--green)}}
</style></head><body>
<nav class="side"><div class="brand">LlamaForge</div>{nav}</nav>
<main>{body}</main></body></html>"""


def _nav(pages, current):
    from collections import OrderedDict
    secs = OrderedDict()
    for p in pages:
        secs.setdefault(p["section"], []).append(p)
    parts = []
    for sec, items in secs.items():
        parts.append('<div class="sec">%s</div>' % html.escape(docs.section_title(sec)))
        for p in items:
            cls = ' class="active"' if p["slug"] == current else ""
            parts.append('<a href="%s.html"%s>%s</a>' % (p["slug"], cls, html.escape(p["title"])))
    return "".join(parts)


def build(out_dir, content=None):
    pages = docs.list_pages()
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for p in pages:
        pg = docs.page(p["slug"])
        body = _md_links_to_html(pg["html"])
        htmlpage = _TEMPLATE.format(title=html.escape(pg["title"]), nav=_nav(pages, p["slug"]), body=body)
        dest = os.path.join(out_dir, p["slug"] + ".html")
        with open(dest, "w", encoding="utf-8") as f:
            f.write(htmlpage)
        written.append(dest)
        # the page's markdown too, for llms.txt readers; its sibling links
        # (slug.md) and image paths (docs/img/...) resolve as they are
        _, md = docs.parse_frontmatter(docs._read(os.path.join(docs.content_dir(), p["slug"] + ".md")))
        with open(os.path.join(out_dir, p["slug"] + ".md"), "w", encoding="utf-8") as f:
            f.write(md.lstrip("\n"))
    # landing page: if no page is named "index", copy the first page as index.html
    index = os.path.join(out_dir, "index.html")
    if pages and not any(p["slug"] == "index" for p in pages):
        shutil.copyfile(os.path.join(out_dir, pages[0]["slug"] + ".html"), index)
        written.append(index)
    llms = os.path.join(_ROOT, "llms.txt")
    if os.path.isfile(llms):
        shutil.copyfile(llms, os.path.join(out_dir, "llms.txt"))
    # copy images
    img_src = os.path.join(docs.content_dir(), "img")
    if os.path.isdir(img_src):
        img_dst = os.path.join(out_dir, "docs", "img")
        os.makedirs(img_dst, exist_ok=True)
        for fn in os.listdir(img_src):
            if fn == ".gitkeep":
                continue
            shutil.copyfile(os.path.join(img_src, fn), os.path.join(img_dst, fn))
    return written


def main():
    out = os.path.join(_ROOT, "site")
    if os.path.isdir(out):
        shutil.rmtree(out)
    written = build(out)
    print("wrote %d pages to %s" % (len(written), out))


if __name__ == "__main__":
    main()
