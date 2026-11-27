"""Render a SimulationResult into one self-contained, offline HTML tearsheet.

The page is the result JSON plus ``assets/renderer.js`` and ``assets/tearsheet.css``
(no CDN, no fonts, no network), so it can be emailed or attached as a single file.
The web UI serves the same two assets, so both surfaces draw identical reports.
"""

from __future__ import annotations

from importlib import resources

from jinja2 import Environment, PackageLoader
from markupsafe import Markup

from velox.result import SimulationResult

ASSETS = ("renderer.js", "tearsheet.css")


def asset(name: str) -> str:
    """Text of a bundled report asset (``renderer.js`` or ``tearsheet.css``)."""
    if name not in ASSETS:
        raise KeyError(name)
    return resources.files("velox.report").joinpath("assets", name).read_text("utf-8")


def _embed_safe(text: str) -> str:
    # never let embedded content close the <script>/<style> element it lives in
    return text.replace("</", "<\\/")


def render_html(result: SimulationResult) -> str:
    env = Environment(loader=PackageLoader("velox.report", "templates"), autoescape=True)
    template = env.get_template("tearsheet.html.j2")
    return template.render(
        title=result.ruleset.name,
        payload=Markup(_embed_safe(result.model_dump_json())),
        renderer=Markup(_embed_safe(asset("renderer.js"))),
        styles=Markup(_embed_safe(asset("tearsheet.css"))),
    )
