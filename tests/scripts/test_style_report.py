"""style_report: summaries, the CSV/PNG outputs, and checkpoint discovery (no torch training)."""

from scripts.style_report import checkpoint_steps, plot, summarize, table


def row(source, step, lines, **kw):
    base = {"source": source, "step": step, "seed": 0, "pieces": 10, "lines": lines, "avg_stack_height": 5.0,
            "max_stack_height": 7, "avg_holes": 1.0, "avg_bumpiness": 3.0, "share_singles": 1.0 if lines else 0.0,
            "share_doubles": 0.0, "share_triples": 0.0, "share_tetrises": 0.0,
            "pieces_per_clear": 10 / lines if lines else None}
    return {**base, **kw}


def test_summarize_skips_no_line_games_for_shares():
    rows = [row("fresh", 100, 0, avg_holes=3.0), row("fresh", 100, 2, share_singles=0.5, share_doubles=0.5),
            row("saved", 100, 1)]
    s = {(x["source"], x["step"]): x for x in summarize(rows)}
    f = s[("fresh", 100)]
    assert f["games"] == 2 and f["avg_holes"] == 2.0
    assert f["share_singles"] == 0.5  # the no-line game doesn't count as "0% singles"
    assert f["pieces_per_clear"] == 5.0  # None ignored
    assert "100" in table(summarize(rows), "fresh")


def test_plot_and_checkpoints(tmp_path):
    rows = [row("saved", st, 1) for st in (10, 100, 1000)] + [row("fresh", st, 2) for st in (10, 100, 1000)]
    plot(rows, summarize(rows), 100, "test", tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 1000
    (tmp_path / "checkpoints").mkdir()
    for name in ("step000000010.pt", "step000000200.pt", "best.pt", "latest.pt"):
        (tmp_path / "checkpoints" / name).write_bytes(b"")
    assert sorted(checkpoint_steps(tmp_path)) == [10, 200]
