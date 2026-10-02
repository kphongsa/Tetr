# Web replay viewer

Plain HTML + JavaScript + canvas (no frameworks, no build step). The page only
**draws** games: every board was computed by the Python engine and exported as
frames (format: `games/tetris/frames.py`).

## Run it locally

```
python -m scripts.build_site                       # export replays -> site/data/ (~40 s, ~11 MB)
python -m http.server 8000 --directory site       # serve the folder
```

Open <http://localhost:8000>. Stop the server with Ctrl+C.

**Why not just double-click `index.html`?** A page opened from disk has a
`file://` address. For security, browsers don't let such a page read other
files with `fetch()`, otherwise any downloaded HTML file could read your disk.
The viewer needs to fetch `data/index.json` and the games, so it must be served
over `http://`. `python -m http.server` is a tiny web server built into Python.

`build_site` options: `--runs lr_decay raw`, `--per-run 30` (more training steps
per run), `--all` (every replay, ~46 MB). The URL remembers what you're
watching (`#a=<game>&b=<game>&n=<piece>`), so you can bookmark a moment.

## Publishing on GitHub Pages later (not set up yet)

GitHub Pages hosts static files (HTML/JS/JSON) for free from a public repo.
What it would need:

1. **The data has to be in git.** `site/data/` is git-ignored today, and the
   replays in `runs/` are too, so GitHub can't build it. Either commit a
   chosen, smaller export (e.g. `--per-run 5`, a few MB), or push it to a
   separate `gh-pages` branch so the main history stays small.
2. **Tell Pages which folder to serve.** Pages serves the repo root or a
   `/docs` folder of a branch, or any folder through a small GitHub Actions
   workflow. Simplest: a `gh-pages` branch whose root is the contents of `site/`.
3. **Turn it on** in the repo's Settings → Pages. The site appears at
   `https://<user>.github.io/<repo>/`. All paths in the viewer are relative
   (`data/...`), so it works under that sub-path unchanged.

Limits worth knowing: files under 100 MB, site under 1 GB. Pages compresses
JSON on the fly (a 1.3 MB game downloads as ~85 KB).
