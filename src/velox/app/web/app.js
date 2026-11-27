/* VELOX web simulator: the form, the data-quality preview and the result view.
 *
 * The server is stateless: every call uploads the trade log again. The result is
 * drawn by renderVelox() from renderer.js, the same code as the offline tearsheet.
 * Strings from the server are inserted with textContent, never innerHTML.
 */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const form = $("sim-form");
  const state = { meta: null, resultText: null, result: null, inspectSeq: 0, timer: null };

  const money = (v) => (v == null || !isFinite(v) ? "–" :
    (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 }));
  const pct = (p) => (p == null ? "–" : (p * 100).toFixed(1) + "%");

  function el(tag, text, cls) {
    const n = document.createElement(tag);
    if (text != null) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  }

  function errorText(body, status) {
    const d = body && body.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {  // FastAPI's own validation errors
      return d.map((e) => `${(e.loc || []).filter((x) => x !== "body").join(".")}: ${e.msg}`).join("; ");
    }
    return `request failed (HTTP ${status})`;
  }

  async function call(url, opts) {
    const res = await fetch(url, opts);
    const text = await res.text();
    let body = null;
    try { body = JSON.parse(text); } catch (_) { /* HTML or empty */ }
    if (!res.ok) throw new Error(errorText(body, res.status));
    return { res, text, body };
  }

  // ── form setup from /api/meta ─────────────────────────────────────────────
  function fillDefaults(defaults) {
    for (const input of form.querySelectorAll("[data-req]")) {
      const v = defaults[input.name];
      if (v == null) continue;
      input.value = Array.isArray(v) ? v.join(", ") : String(v);
    }
  }

  function setupMeta(meta) {
    state.meta = meta;
    const preset = $("preset");
    for (const p of meta.presets) {
      const o = el("option", `${p.name} · ${money(p.initial_balance)}`);
      o.value = p.id;
      preset.append(o);
    }
    const intra = $("intratrade");
    for (const m of meta.intratrade_models) {
      const o = el("option", m);
      o.value = m;
      intra.append(o);
    }
    fillDefaults(meta.defaults);
    form.elements.n_paths.max = meta.limits.max_paths;
    onPreset();
    onFormat();
  }

  function currentPreset() {
    return state.meta.presets.find((p) => p.id === $("preset").value);
  }

  function onPreset() {
    const p = currentPreset();
    if (!p) return;
    const parts = [
      p.phases.join(" → ") + (p.funded ? ` → ${p.funded}` : ""),
      `trading day rolls at ${p.rollover} ${p.timezone}`,
      `rules verified ${p.verified_at}`,
    ];
    if (p.needs_verification.length) parts.push(`${p.needs_verification.length} open rule questions`);
    $("preset-hint").textContent = parts.join(" · ");
    $("balance").placeholder = String(p.initial_balance);
    form.elements.activation_fee.placeholder = String(p.activation_fee);
  }

  function onFormat() {
    const f = state.meta.formats.find((x) => x.id === $("format").value);
    $("commission-field").hidden = !f.gross_pnl;
    if (!f.gross_pnl) $("commission").value = "0";
    $("tz-hint").textContent = f.needs_timezone
      ? "Required: this export has no UTC offset. Use the broker server's or report's timezone."
      : "Needed when timestamps have no UTC offset. Broker server time is rarely UTC.";
  }

  // ── multipart body shared by /api/inspect and /api/simulate ───────────────
  function logForm() {
    const fd = new FormData();
    fd.append("file", $("file").files[0]);
    fd.append("format", $("format").value);
    fd.append("preset", $("preset").value);
    const pf = $("preset-file").files[0];
    if (pf) fd.append("preset_file", pf);
    if ($("tz").value.trim()) fd.append("tz", $("tz").value.trim());
    if (!$("commission-field").hidden && $("commission").value) {
      fd.append("commission_per_unit", $("commission").value);
    }
    if ($("balance").value) fd.append("balance", $("balance").value);
    return fd;
  }

  function settings() {
    const req = {};
    const fee = $("fee");
    fee.setAttribute("aria-invalid", "false");
    if (fee.value === "" || !(+fee.value >= 0)) {
      fee.setAttribute("aria-invalid", "true");
      fee.focus();
      throw new Error("Enter the challenge fee you pay (0 if it is free).");
    }
    req.challenge_fee = +fee.value;
    for (const input of form.querySelectorAll("[data-req]")) {
      const raw = input.value.trim();
      if (raw === "") continue;  // blank = server default
      if (input.tagName === "SELECT") req[input.name] = raw;
      else if ("list" in input.dataset) {
        req[input.name] = raw.split(",").map((x) => x.trim()).filter(Boolean).map(Number);
      } else if ("int" in input.dataset) req[input.name] = parseInt(raw, 10);
      else req[input.name] = Number(raw);
    }
    return req;
  }

  // ── data-quality preview ──────────────────────────────────────────────────
  let inspectDebounce = null;
  function scheduleInspect() {
    clearTimeout(inspectDebounce);
    inspectDebounce = setTimeout(inspect, 250);
  }

  async function inspect() {
    const box = $("inspect");
    if (!$("file").files[0]) { box.hidden = true; return; }
    const seq = ++state.inspectSeq;
    const fd = logForm();
    fd.append("intratrade", form.elements.intratrade.value || "auto");
    box.hidden = false;
    box.className = "inspect";
    box.replaceChildren(el("p", "Checking the trade log…"));
    try {
      const { body } = await call("/api/inspect", { method: "POST", body: fd });
      if (seq !== state.inspectSeq) return;  // a newer check is running
      const q = body.quality, t = body.trades, d = body.days;
      const first = q.first_trade.slice(0, 10), last = q.last_trade.slice(0, 10);
      box.replaceChildren(
        el("p", `${q.n_trades} trades over ${q.n_trading_days} ${body.ruleset.name} trading days ` +
                `(${first} to ${last}). Total ${money(t.total)}, win rate ${pct(t.win_rate)}, ` +
                `best day ${money(d.best)}, worst day ${money(d.worst)}. ` +
                `Intra-trade model: ${q.intratrade_model}.`));
      if (q.warnings.length) {
        const ul = el("ul");
        q.warnings.forEach((w) => ul.append(el("li", w)));
        box.append(ul);
      }
    } catch (e) {
      if (seq !== state.inspectSeq) return;
      box.className = "inspect bad";
      box.replaceChildren(el("p", e.message));
    }
  }

  // ── simulate ──────────────────────────────────────────────────────────────
  function busy(on) {
    $("run").disabled = on;
    clearInterval(state.timer);
    if (!on) return;
    const t0 = performance.now();
    const tick = () => {
      $("status").textContent = `Simulating… ${((performance.now() - t0) / 1000).toFixed(0)} s`;
    };
    tick();
    state.timer = setInterval(tick, 500);
  }

  function showError(message) {
    const box = $("error");
    box.textContent = message;
    box.hidden = !message;
  }

  async function run(ev) {
    ev.preventDefault();
    showError("");
    if (!$("file").files[0]) { showError("Choose a trade log file first."); return; }
    let req;
    try { req = settings(); } catch (e) { showError(e.message); return; }
    const fd = logForm();
    fd.append("request", JSON.stringify(req));
    busy(true);
    const t0 = performance.now();
    try {
      const { text, body } = await call("/api/simulate", { method: "POST", body: fd });
      state.resultText = text;
      state.result = body;
      $("results").hidden = false;
      renderVelox($("velox-root"), body);
      $("status").textContent =
        `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s · ` +
        `${body.grid.length} sizes × ${body.config.n_paths.toLocaleString()} paths`;
      $("results").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      $("status").textContent = "";
      showError(e.message);
    } finally {
      busy(false);
    }
  }

  // ── downloads ─────────────────────────────────────────────────────────────
  function save(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), { href: url, download: filename });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function downloadHtml() {
    if (!state.resultText) return;
    showError("");
    try {
      const res = await fetch("/api/report", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: state.resultText });
      if (!res.ok) throw new Error(errorText(await res.json().catch(() => null), res.status));
      const cd = res.headers.get("Content-Disposition") || "";
      const name = (cd.match(/filename="([^"]+)"/) || [])[1] || "velox-tearsheet.html";
      save(await res.blob(), name);
    } catch (e) {
      showError(e.message);
    }
  }

  function downloadJson() {
    if (!state.resultText) return;
    save(new Blob([state.resultText], { type: "application/json" }),
         `velox-${state.result.ruleset.id}-result.json`);
  }

  // ── wiring ────────────────────────────────────────────────────────────────
  form.addEventListener("submit", run);
  $("preset").addEventListener("change", () => { onPreset(); scheduleInspect(); });
  $("format").addEventListener("change", () => { onFormat(); scheduleInspect(); });
  for (const id of ["file", "tz", "commission", "balance", "preset-file", "intratrade"]) {
    $(id).addEventListener("change", scheduleInspect);
  }
  $("dl-html").addEventListener("click", downloadHtml);
  $("dl-json").addEventListener("click", downloadJson);

  call("/api/meta")
    .then(({ body }) => setupMeta(body))
    .catch((e) => showError(`Could not reach the VELOX server: ${e.message}`));
})();
