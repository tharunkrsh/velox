"""Render a SimulationResult into one self-contained, offline HTML tearsheet.

The page is the result JSON plus ``assets/renderer.js`` (no CDN, no fonts, no
network), so it can be emailed or attached as a single file.
"""

from __future__ import annotations

from importlib import resources

from jinja2 import Environment, PackageLoader
from markupsafe import Markup

from velox.report.schema import SimulationResult


def _script_safe(text: str) -> str:
    # never let embedded content close the <script> element it lives in
    return text.replace("</", "<\\/")


def render_html(result: SimulationResult) -> str:
    env = Environment(loader=PackageLoader("velox.report", "templates"),
                      autoescape=True)
    renderer = resources.files("velox.report").joinpath("assets/renderer.js").read_text("utf-8")
    template = env.get_template("tearsheet.html.j2")
    return template.render(
        title=result.ruleset.name,
        payload=Markup(_script_safe(result.model_dump_json())),
        renderer=Markup(_script_safe(renderer)),
    )
