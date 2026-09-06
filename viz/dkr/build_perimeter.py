"""Inject dkr_map.json into dkr_perimeter.html -> dkr_perimeter_built.html."""
from pathlib import Path
d = Path(__file__).parent
html = (d / "dkr_perimeter.html").read_text(encoding="utf-8")
js = (d / "dkr_map.json").read_text(encoding="utf-8")
assert html.count("/*MAPDATA*/null") == 1, "placeholder missing"
assert 'aim="split"' in html, "default aim not split"
out = html.replace("/*MAPDATA*/null", js)
(d / "dkr_perimeter_built.html").write_text(out, encoding="utf-8")
print(f"rebuilt OK: {len(out)//1024} KB, default aim = split")
