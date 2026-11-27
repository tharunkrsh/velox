"""Stateless HTTP API and the local web simulator.

Every request carries everything it needs (the trade log, the preset, the
settings), and nothing is stored between requests: uploads are parsed in
memory and never written to disk. The simulation endpoint calls the same
:func:`velox.app.simulation.run_simulation` as ``velox simulate``, and returns the
same :class:`SimulationResult` JSON, byte for byte. The web UI draws it with the
same ``renderer.js`` and ``tearsheet.css`` as the offline HTML tearsheet.

Endpoints:
    GET  /                  web UI
    GET  /assets/{name}     UI and report assets
    GET  /api/health        liveness
    GET  /api/meta          presets, formats, request defaults, limits
    POST /api/inspect       multipart trade log -> data-quality summary
    POST /api/simulate      multipart trade log + request JSON -> SimulationResult
    POST /api/report        SimulationResult JSON -> downloadable HTML tearsheet

Run it with ``velox serve``, or ``uvicorn --factory velox.app.api:create_app``.
There is no authentication: it is meant for localhost.
"""

from __future__ import annotations

import re
import threading
from importlib import resources
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_core import to_jsonable_python

import velox
from velox.app.simulation import SimulationRequest, run_simulation
from velox.domain.daybook import build_daybook
from velox.domain.rules.spec import RulesetSpec
from velox.domain.sizing import IntratradeModel
from velox.domain.stats import PnlSummary, summarize
from velox.domain.trades import TradeLog
from velox.ingest import FORMATS, GROSS_PNL_FORMATS, load_trades
from velox.ingest.quality import DataQualityReport, assess
from velox.presets import list_presets, parse_preset
from velox.report.html import render_html
from velox.result import SCHEMA_VERSION, SimulationResult


class ApiLimits(BaseModel):
    """Guards against requests that would tie the machine up for minutes."""

    model_config = ConfigDict(frozen=True)

    max_upload_bytes: int = 10 * 1024 * 1024
    max_paths: int = 50_000
    max_sizes: int = 121              # multipliers on the size grid
    max_haircuts: int = 4
    max_horizon_days: int = 500       # max_eval_days + funded_days
    max_concurrent_simulations: int = 2


class PresetInfo(BaseModel):
    id: str
    name: str
    version: str
    verified_at: str
    initial_balance: float
    phases: list[str]
    funded: str | None
    activation_fee: float
    timezone: str
    rollover: str
    needs_verification: list[str]


class FormatInfo(BaseModel):
    id: str
    gross_pnl: bool          # P&L is gross of commission, so commission_per_unit applies
    needs_timezone: bool     # timestamps never carry a UTC offset


class Meta(BaseModel):
    velox_version: str
    schema_version: str
    presets: list[PresetInfo]
    formats: list[FormatInfo]
    intratrade_models: list[str]
    defaults: dict
    limits: ApiLimits


class InspectResult(BaseModel):
    ruleset: PresetInfo
    quality: DataQualityReport
    trades: PnlSummary
    days: PnlSummary


# static files the UI may fetch, by public name -> (package, path)
_ASSETS: dict[str, tuple[str, str, str]] = {
    "app.js": ("velox.app", "web/app.js", "text/javascript"),
    "app.css": ("velox.app", "web/app.css", "text/css"),
    "renderer.js": ("velox.report", "assets/renderer.js", "text/javascript"),
    "tearsheet.css": ("velox.report", "assets/tearsheet.css", "text/css"),
}

_CSP = ("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'")

TradeLogFile = Annotated[UploadFile, File(description="Trade log CSV/TSV export")]
PresetId = Annotated[str, Form(description="Bundled preset id")]
PresetFile = Annotated[UploadFile | None, File(description="Custom preset YAML (overrides id)")]
FormatField = Annotated[Literal["generic", "mt5", "tradovate"], Form()]
TzField = Annotated[str | None, Form(description="IANA zone of naive timestamps")]
CommissionField = Annotated[float, Form(ge=0)]
BalanceField = Annotated[float | None, Form(gt=0)]
IntraField = Annotated[IntratradeModel, Form()]


def _preset_info(spec: RulesetSpec) -> PresetInfo:
    return PresetInfo(
        id=spec.id, name=spec.name, version=spec.version, verified_at=str(spec.verified_at),
        initial_balance=spec.initial_balance, phases=[p.name for p in spec.phases],
        funded=spec.funded.name if spec.funded else None,
        activation_fee=spec.funded.activation_fee if spec.funded else 0.0,
        timezone=spec.calendar.timezone, rollover=f"{spec.calendar.rollover:%H:%M}",
        needs_verification=list(spec.needs_verification),
    )


def _unprocessable(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=message)


def _validation_message(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(parts)


def create_app(limits: ApiLimits | None = None) -> FastAPI:
    limits = limits or ApiLimits()
    bundled = {s.id: s for s in list_presets()}
    slots = threading.BoundedSemaphore(limits.max_concurrent_simulations)

    api = FastAPI(title="VELOX", version=velox.__version__,
                  summary="Prop-firm challenge survival & position-sizing simulator",
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")

    # ── request plumbing ────────────────────────────────────────────────────────

    def read_upload(upload: UploadFile, what: str) -> bytes:
        data = upload.file.read(limits.max_upload_bytes + 1)
        if len(data) > limits.max_upload_bytes:
            raise HTTPException(413, f"{what} is larger than "
                                     f"{limits.max_upload_bytes // (1024 * 1024)} MB")
        if not data:
            raise _unprocessable(f"{what} is empty")
        return data

    def resolve_ruleset(preset: str, preset_file: UploadFile | None,
                        balance: float | None) -> RulesetSpec:
        if preset_file is not None and preset_file.filename:
            try:
                spec = parse_preset(read_upload(preset_file, "the preset file"))
            except ValidationError as e:
                raise _unprocessable(f"invalid preset: {_validation_message(e)}") from e
            except ValueError as e:
                raise _unprocessable(str(e)) from e
        else:
            # bundled ids only: never resolve a server-side file path from a request
            if preset not in bundled:
                raise _unprocessable(
                    f"unknown preset '{preset}' (bundled: {', '.join(sorted(bundled))})")
            spec = bundled[preset]
        return spec.with_balance(balance) if balance else spec

    def parse_log(file: UploadFile, fmt: str, tz: str | None, commission: float) -> TradeLog:
        data = read_upload(file, "the trade log")
        try:
            return load_trades(data, fmt, tz or None,  # type: ignore[arg-type]
                               commission_per_unit=commission)
        except ValueError as e:  # TradeLogError included
            raise _unprocessable(str(e)) from e

    def parse_request(raw: str) -> SimulationRequest:
        try:
            req = SimulationRequest.model_validate_json(raw)
        except ValidationError as e:
            raise _unprocessable(f"invalid settings: {_validation_message(e)}") from e
        checks = [
            (req.n_paths <= limits.max_paths, f"n_paths is capped at {limits.max_paths:,}"),
            (len(req.multipliers()) <= limits.max_sizes,
             f"the size grid has {len(req.multipliers())} multipliers; the cap is "
             f"{limits.max_sizes} (raise k_step or narrow k_min..k_max)"),
            (len(req.haircuts) <= limits.max_haircuts,
             f"at most {limits.max_haircuts} edge haircuts"),
            (req.max_eval_days + req.funded_days <= limits.max_horizon_days,
             f"max_eval_days + funded_days is capped at {limits.max_horizon_days}"),
        ]
        for ok, message in checks:
            if not ok:
                raise _unprocessable(message)
        return req

    # ── UI ──────────────────────────────────────────────────────────────────────

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        page = resources.files("velox.app").joinpath("web", "index.html").read_text("utf-8")
        return HTMLResponse(page, headers={"Content-Security-Policy": _CSP,
                                           "Cache-Control": "no-cache"})

    @api.get("/assets/{name}", include_in_schema=False)
    def static_asset(name: str) -> Response:
        if name not in _ASSETS:
            raise HTTPException(404, "not found")
        package, path, media_type = _ASSETS[name]
        text = resources.files(package).joinpath(*path.split("/")).read_text("utf-8")
        return Response(text, media_type=media_type, headers={"Cache-Control": "no-cache"})

    # ── API ─────────────────────────────────────────────────────────────────────

    @api.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "velox_version": velox.__version__}

    @api.get("/api/meta", response_model=Meta)
    def meta() -> Meta:
        # challenge_fee is required, so it has no default and is left out
        defaults = to_jsonable_python({
            name: f.get_default(call_default_factory=True)
            for name, f in SimulationRequest.model_fields.items() if not f.is_required()})
        return Meta(
            velox_version=velox.__version__, schema_version=SCHEMA_VERSION,
            presets=[_preset_info(s) for s in bundled.values()],
            formats=[FormatInfo(id=f, gross_pnl=f in GROSS_PNL_FORMATS,
                                needs_timezone=f != "generic") for f in FORMATS],
            intratrade_models=[m.value for m in IntratradeModel],
            defaults=defaults, limits=limits,
        )

    @api.post("/api/inspect", response_model=InspectResult)
    def inspect(file: TradeLogFile, preset: PresetId = "ftmo-2step",
                preset_file: PresetFile = None, format: FormatField = "generic",
                tz: TzField = None, commission_per_unit: CommissionField = 0.0,
                balance: BalanceField = None,
                intratrade: IntraField = IntratradeModel.AUTO) -> InspectResult:
        spec = resolve_ruleset(preset, preset_file, balance)
        log = parse_log(file, format, tz, commission_per_unit)
        book = build_daybook(log, spec.calendar, intratrade)
        return InspectResult(
            ruleset=_preset_info(spec), quality=assess(log, spec.calendar, intratrade),
            trades=summarize(log.pnl), days=summarize(book.daily_pnl()),
        )

    @api.post("/api/simulate", response_model=SimulationResult)
    def simulate(file: TradeLogFile,
                 request: Annotated[str, Form(description="SimulationRequest as JSON")],
                 preset: PresetId = "ftmo-2step", preset_file: PresetFile = None,
                 format: FormatField = "generic", tz: TzField = None,
                 commission_per_unit: CommissionField = 0.0,
                 balance: BalanceField = None) -> Response:
        spec = resolve_ruleset(preset, preset_file, balance)
        req = parse_request(request)
        log = parse_log(file, format, tz, commission_per_unit)
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "the simulator is busy with other runs; try again shortly")
        try:
            result = run_simulation(log, spec, req)
        finally:
            slots.release()
        # serialise exactly as `velox simulate --json` does, so the two are comparable
        return Response(result.model_dump_json(), media_type="application/json")

    @api.post("/api/report", response_class=HTMLResponse)
    def report(result: SimulationResult) -> HTMLResponse:
        slug = re.sub(r"[^a-z0-9]+", "-", result.ruleset.id.lower()).strip("-") or "velox"
        return HTMLResponse(render_html(result), headers={
            "Content-Disposition": f'attachment; filename="velox-{slug}-tearsheet.html"'})

    return api
