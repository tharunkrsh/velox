/* VELOX tearsheet renderer.
 *
 * renderVelox(root, result) draws a SimulationResult (schema 1.0) into `root`.
 * Pure DOM + inline SVG, no dependencies and no network access. The HTML report embeds
 * this file next to the result JSON; the Phase 3 web UI reuses it unchanged.
 * All strings from the payload are inserted with textContent, never innerHTML.
 */
(function (global) {
  "use strict";
  const SVG = "http://www.w3.org/2000/svg";

  // ── formatting ──────────────────────────────────────────────────────────────
  const pct = (p, d = 1) => (p == null ? "–" : (p * 100).toFixed(d) + "%");
  const money = (v, signed = false) => {
    if (v == null || !isFinite(v)) return "–";
    const s = Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 });
    const sign = v < 0 ? "−" : signed && v > 0 ? "+" : "";
    return sign + "$" + s;
  };
  const compact = (v) => {
    const a = Math.abs(v);
    const sign = v < 0 ? "−" : "";
    if (a >= 1e6) return sign + "$" + (a / 1e6).toFixed(a % 1e6 ? 1 : 0) + "M";
    if (a >= 1e4) return sign + "$" + (a / 1e3).toFixed(0) + "K";
    return sign + "$" + a.toLocaleString("en-US", { maximumFractionDigits: 0 });
  };
  const kfmt = (k) => "×" + (+k).toFixed(2).replace(/\.?0+$/, "");

  // ── dom helpers ─────────────────────────────────────────────────────────────
  function el(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else n.setAttribute(k, v);
    }
    for (const c of kids) if (c != null) n.append(c);
    return n;
  }
  function s(tag, attrs = {}) {
    const n = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  }
  function niceTicks(lo, hi, count = 5) {
    if (lo === hi) { lo -= 1; hi += 1; }
    const span = hi - lo;
    const step0 = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => span / x <= count) || 10 * mag;
    const t = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
    return t;
  }

  // ── chart frame with crosshair tooltip ──────────────────────────────────────
  const W = 640, H = 240, M = { l: 64, r: 92, t: 12, b: 34 };

  function frame({ xs, xDomain, yDomain, yFmt, xFmt, xTicks, yZero }) {
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" });
    const [x0, x1] = xDomain, [y0, y1] = yDomain;
    const X = (v) => M.l + ((v - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
    const Y = (v) => H - M.b - ((v - y0) / (y1 - y0 || 1)) * (H - M.t - M.b);
    const g = s("g", { class: "axes" });
    for (const t of niceTicks(y0, y1)) {
      if (t < y0 - 1e-9 || t > y1 + 1e-9) continue;
      g.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), class: "grid" }));
      const lab = s("text", { x: M.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "tick" });
      lab.textContent = yFmt(t);
      g.append(lab);
    }
    if (yZero && y0 < 0 && y1 > 0)
      g.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(0), y2: Y(0), class: "baseline" }));
    g.append(s("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: "baseline" }));
    for (const t of xTicks) {
      const lab = s("text", { x: X(t), y: H - M.b + 18, "text-anchor": "middle", class: "tick" });
      lab.textContent = xFmt(t);
      g.append(lab);
    }
    svg.append(g);
    return { svg, X, Y };
  }

  function attachCrosshair(wrap, svg, X, xs, rowsAt) {
    const tip = el("div", { class: "tip", role: "status" });
    tip.hidden = true;
    const cross = s("line", { y1: M.t, y2: H - M.b, class: "cross" });
    cross.style.display = "none";
    svg.append(cross);
    const hit = s("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b,
                            fill: "transparent", tabindex: "0" });
    svg.append(hit);
    let idx = 0;
    function show(i) {
      idx = Math.max(0, Math.min(xs.length - 1, i));
      const x = X(xs[idx]);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.style.display = "";
      tip.replaceChildren(...rowsAt(idx));
      tip.hidden = false;
      const box = svg.getBoundingClientRect(), scale = box.width / W;
      const left = x * scale;
      tip.style.left = Math.min(Math.max(left + 12, 0), box.width - 190) + "px";
      tip.style.top = "8px";
    }
    function nearest(evt) {
      const box = svg.getBoundingClientRect();
      const vx = ((evt.clientX - box.left) / box.width) * W;
      let best = 0, bd = Infinity;
      xs.forEach((v, i) => { const d = Math.abs(X(v) - vx); if (d < bd) { bd = d; best = i; } });
      return best;
    }
    hit.addEventListener("pointermove", (e) => show(nearest(e)));
    hit.addEventListener("pointerleave", () => { tip.hidden = true; cross.style.display = "none"; });
    hit.addEventListener("focus", () => show(idx));
    hit.addEventListener("blur", () => { tip.hidden = true; cross.style.display = "none"; });
    hit.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") { show(idx + 1); e.preventDefault(); }
      if (e.key === "ArrowLeft") { show(idx - 1); e.preventDefault(); }
    });
    wrap.append(tip);
  }

  function tipRow(value, label, cssVar) {
    const key = el("span", { class: "key" });
    if (cssVar) key.style.background = `var(${cssVar})`;
    return el("div", { class: "tip-row" }, key, el("strong", { text: value }), el("span", { text: label }));
  }

  function legend(items) {
    return el("div", { class: "legend" }, ...items.map(([name, cssVar, kind]) =>
      el("span", { class: "legend-item" },
         el("span", { class: kind === "band" ? "swatch band" : "swatch line" ,
                      style: `--c: var(${cssVar})` }),
         el("span", { text: name }))));
  }

  function card(title, subtitle, ...body) {
    return el("section", { class: "card" },
      el("h2", { text: title }), subtitle ? el("p", { class: "sub", text: subtitle }) : null, ...body);
  }

  // ── line chart over the multiplier grid, with CI bands and markers ─────────
  function kChart({ grid, series, yFmt, yZero, markers }) {
    const xs = grid.map((g) => g.k);
    const all = series.flatMap((sr) => grid.flatMap((g) => [sr.lo ? sr.lo(g) : sr.v(g), sr.hi ? sr.hi(g) : sr.v(g)]));
    let y0 = Math.min(...all), y1 = Math.max(...all);
    if (yZero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
    const pad = (y1 - y0) * 0.06 || 1;
    const ticks = niceTicks(y0 - pad, y1 + pad);
    y0 = Math.min(ticks[0], y0); y1 = Math.max(ticks[ticks.length - 1], y1);
    const xTicks = niceTicks(xs[0], xs[xs.length - 1], 7).filter((t) => t >= xs[0] - 1e-9);
    const { svg, X, Y } = frame({ xs, xDomain: [xs[0], xs[xs.length - 1]], yDomain: [y0, y1],
                                  yFmt, xFmt: kfmt, xTicks, yZero });
    const wrap = el("div", { class: "chart-wrap" });
    for (const sr of series) {
      if (sr.lo) {
        const up = grid.map((g) => `${X(g.k)},${Y(sr.hi(g))}`);
        const dn = grid.map((g) => `${X(g.k)},${Y(sr.lo(g))}`).reverse();
        svg.append(s("polygon", { points: up.concat(dn).join(" "), class: "ci-band",
                                  style: `fill: var(${sr.color})` }));
      }
      svg.append(s("polyline", { points: grid.map((g) => `${X(g.k)},${Y(sr.v(g))}`).join(" "),
                                 class: "line", style: `stroke: var(${sr.color})` }));
      const last = grid[grid.length - 1];
      if (sr.endLabel) {
        const t = s("text", { x: X(last.k) + 8, y: Y(sr.v(last)) + 4, class: "end-label" });
        t.textContent = sr.endLabel;
        svg.append(t);
      }
    }
    for (const m of markers || []) {
      const g = grid.find((x) => x.k === m.k);
      if (!g) continue;
      const sr = series[0];
      svg.append(s("circle", { cx: X(g.k), cy: Y(sr.v(g)), r: 5, class: "dot",
                               style: `fill: var(${sr.color})` }));
      const t = s("text", { x: X(g.k), y: Y(sr.v(g)) - 12, "text-anchor": "middle", class: "mark-label" });
      t.textContent = m.label;
      svg.append(t);
    }
    attachCrosshair(wrap, svg, X, xs, (i) => [
      el("div", { class: "tip-head", text: "Size " + kfmt(grid[i].k) }),
      ...series.map((sr) => tipRow(sr.tip(grid[i]), sr.name, sr.color)),
    ]);
    wrap.prepend(svg);
    return wrap;
  }

  // ── fan chart: balance quantiles by simulated day ───────────────────────────
  function fanChart(fan, initial, target, yDomain) {
    const q = fan.quantiles, xs = fan.days;
    const xTicks = niceTicks(0, xs[xs.length - 1], 6);
    const { svg, X, Y } = frame({ xs, xDomain: [0, xs[xs.length - 1]], yDomain, yFmt: compact,
                                  xFmt: (d) => String(d), xTicks });
    const wrap = el("div", { class: "chart-wrap" });
    const band = (lo, hi, cls) => {
      const up = xs.map((d, i) => `${X(d)},${Y(hi[i])}`);
      const dn = xs.map((d, i) => `${X(d)},${Y(lo[i])}`).reverse();
      svg.append(s("polygon", { points: up.concat(dn).join(" "), class: cls }));
    };
    for (const p of fan.samples) {
      svg.append(s("polyline", { points: xs.map((d, i) => `${X(d)},${Y(p[i])}`).join(" "),
                                 class: "sample" }));
    }
    band(q.p5, q.p95, "fan-outer");
    band(q.p25, q.p75, "fan-inner");
    svg.append(s("polyline", { points: xs.map((d, i) => `${X(d)},${Y(q.p50[i])}`).join(" "),
                               class: "line", style: "stroke: var(--series-1)" }));
    const ref = (v, label) => {
      if (v == null || v < yDomain[0] || v > yDomain[1]) return;
      svg.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), class: "ref" }));
      const t = s("text", { x: W - M.r + 6, y: Y(v) + 4, class: "end-label" });
      t.textContent = label;
      svg.append(t);
    };
    ref(initial, "start");
    ref(target, "target");
    attachCrosshair(wrap, svg, X, xs, (i) => [
      el("div", { class: "tip-head", text: "Day " + xs[i] }),
      tipRow(compact(q.p95[i]), "95th percentile", null),
      tipRow(compact(q.p75[i]), "75th percentile", null),
      tipRow(compact(q.p50[i]), "median", "--series-1"),
      tipRow(compact(q.p25[i]), "25th percentile", null),
      tipRow(compact(q.p5[i]), "5th percentile", null),
    ]);
    wrap.prepend(svg);
    return wrap;
  }

  // ── 100% segmented bars: how paths end ──────────────────────────────────────
  const OUTCOME_PARTS = [
    ["Passed", "--series-1", (g) => g.p_pass.p],
    ["Daily loss breach", "--series-2", (g) => g.p_fail_daily_loss.p],
    ["Max loss breach", "--series-3", (g) => g.p_fail_max_loss.p],
    ["Still running at cap", "--series-4", (g) => g.p_timeout.p],
  ];
  function outcomeBars(rows) {
    const wrap = el("div", { class: "bars" });
    const tip = el("div", { class: "tip" }); tip.hidden = true;
    for (const [label, g] of rows) {
      const bar = el("div", { class: "seg-bar" });
      for (const [name, cssVar, f] of OUTCOME_PARTS) {
        const v = f(g);
        if (v <= 0) continue;
        const seg = el("div", { class: "seg", tabindex: "0", style: `flex-basis:${v * 100}%; background: var(${cssVar})` });
        const showTip = () => {
          tip.replaceChildren(el("div", { class: "tip-head", text: label }), tipRow(pct(v), name, cssVar));
          tip.hidden = false;
          const r = seg.getBoundingClientRect(), w = wrap.getBoundingClientRect();
          tip.style.left = Math.min(r.left - w.left, w.width - 190) + "px";
          tip.style.top = (r.bottom - w.top + 6) + "px";
        };
        seg.addEventListener("pointerenter", showTip);
        seg.addEventListener("focus", showTip);
        seg.addEventListener("pointerleave", () => (tip.hidden = true));
        seg.addEventListener("blur", () => (tip.hidden = true));
        if (v >= 0.12) seg.append(el("span", { class: "seg-label", text: pct(v, 0) }));
        bar.append(seg);
      }
      wrap.append(el("div", { class: "bar-row" }, el("div", { class: "bar-label", text: label }), bar));
    }
    wrap.append(tip);
    return wrap;
  }

  function table(head, rows, highlight) {
    const t = el("table");
    t.append(el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))));
    const tb = el("tbody");
    rows.forEach((r, i) => tb.append(el("tr", highlight && highlight(i) ? { class: "hl" } : {},
      ...r.map((c) => el("td", { text: c })))));
    t.append(tb);
    return el("div", { class: "table-wrap" }, t);
  }

  function tile(label, value, note) {
    return el("div", { class: "tile" }, el("div", { class: "tile-label", text: label }),
      el("div", { class: "tile-value", text: value }), note ? el("div", { class: "tile-note", text: note }) : null);
  }

  // ── page ────────────────────────────────────────────────────────────────────
  function renderVelox(root, r) {
    root.replaceChildren();
    const rec = r.recommendation, cfg = r.config, rs = r.ruleset;
    const byK = new Map(r.grid.map((g) => [g.k, g]));
    const cur = byK.get(rec.k_current), best = byK.get(rec.k);

    root.append(el("header", { class: "head" },
      el("div", { class: "eyebrow", text: "VELOX challenge simulation" }),
      el("h1", { text: rs.name }),
      el("p", { class: "sub", text:
        `${money(rs.initial_balance)} account · fee ${money(cfg.challenge_fee)}` +
        (cfg.activation_fee ? ` + activation ${money(cfg.activation_fee)}` : "") +
        ` · ${cfg.n_paths.toLocaleString()} paths per size · ${cfg.resampler} bootstrap · seed ${cfg.seed}` +
        ` · rules verified ${rs.verified_at}` })));

    root.append(el("div", { class: "tiles" },
      tile("Pass probability at your size", pct(cur.p_pass.p),
           `95% CI ${pct(cur.p_pass.lo)}–${pct(cur.p_pass.hi)}`),
      tile("Recommended size", kfmt(rec.k),
           rec.risk_per_trade_recommended != null
             ? `≈ ${money(rec.risk_per_trade_recommended)} risk per trade (now ${money(rec.risk_per_trade_current)})`
             : `of your historical position size`),
      tile("Expected value per attempt", money(best.ev, true),
           `at ${kfmt(rec.k)}; ${money(cur.ev, true)} at your size`),
      tile("Break-even funded payout", money(cur.breakeven_funded_payout),
           "needed on average to recoup fees at your size")));

    root.append(card("Why this size", null,
      el("ul", { class: "rationale" }, ...rec.rationale.map((t) => el("li", { text: t })))));

    root.append(card("Pass probability by position size",
      "Share of simulated attempts that pass every evaluation phase, with 95% Wilson intervals.",
      kChart({ grid: r.grid, yFmt: (v) => pct(v, 0),
        series: [{ name: "pass probability", color: "--series-1", v: (g) => g.p_pass.p,
                   lo: (g) => g.p_pass.lo, hi: (g) => g.p_pass.hi,
                   tip: (g) => `${pct(g.p_pass.p)} (${pct(g.p_pass.lo)}–${pct(g.p_pass.hi)})` }],
        markers: [{ k: rec.k_current, label: "you" }, ...(rec.k !== rec.k_current ? [{ k: rec.k, label: "best EV" }] : [])] })));

    root.append(card("Expected value per attempt by position size",
      "Simulated funded payouts (after split) minus challenge and activation fees, with 95% intervals.",
      kChart({ grid: r.grid, yFmt: compact, yZero: true,
        series: [{ name: "expected value", color: "--series-1", v: (g) => g.ev,
                   lo: (g) => g.ev_lo, hi: (g) => g.ev_hi,
                   tip: (g) => `${money(g.ev, true)} · break-even payout ${money(g.breakeven_funded_payout)}` }],
        markers: [{ k: rec.k, label: "best" }] })));

    // fans share one y-domain so current vs recommended compare honestly
    const allQ = r.fans.flatMap((f) => [...f.quantiles.p5, ...f.quantiles.p95, ...f.samples.flat()]);
    const target = rs.phase_targets.length ? rs.initial_balance + rs.phase_targets[0] : null;
    const yd = [Math.min(...allQ), Math.max(...allQ)];
    const fanGrid = el("div", { class: "fans" }, ...r.fans.map((f) =>
      el("div", {}, el("h3", { text: `${f.label} (${kfmt(f.k)})` }),
        fanChart(f, rs.initial_balance, target, yd))));
    root.append(card("Evaluation account balance",
      "Median, 25–75% and 5–95% bands of the evaluation balance by trading day, with sample paths. " +
      "Balances reset when a phase is passed; paths that end stay at their final balance.",
      legend([["median", "--series-1", "line"], ["25–75%", "--fan-inner", "band"], ["5–95%", "--fan-outer", "band"]]),
      fanGrid));

    const rows = [["your size " + kfmt(cur.k), cur]];
    if (best.k !== cur.k) rows.push(["recommended " + kfmt(best.k), best]);
    root.append(card("How attempts end",
      `Share of simulated attempts by outcome. "Still running" = neither passed nor failed within ${cfg.max_eval_days} trading days.`,
      legend(OUTCOME_PARTS.map(([n, c]) => [n, c, "band"])), outcomeBars(rows),
      el("p", { class: "note", text:
        `Consistency or minimum-days rules delayed ${pct(cur.p_consistency_delay.p)} of attempts at your size` +
        (cur.mean_delay_days != null ? ` (by ${cur.mean_delay_days.toFixed(1)} trading days on average).` : ".") })));

    const stressSeries = [
      { name: "no haircut", color: "--series-1", data: r.grid.map((g) => ({ k: g.k, ev: g.ev })) },
      ...r.stress.map((sc, i) => ({ name: `${pct(sc.haircut, 0)} edge haircut`, color: `--series-${i + 2}`,
                                    data: sc.grid.map((p) => ({ k: p.k, ev: p.ev })) })),
    ];
    const ks = r.stress.length ? r.stress[0].grid.map((p) => p.k) : r.grid.map((g) => g.k);
    const stressGrid = ks.map((k) => {
      const o = { k };
      stressSeries.forEach((sr, i) => { const p = sr.data.find((d) => d.k === k); o["s" + i] = p ? p.ev : null; });
      return o;
    }).filter((o) => stressSeries.every((_, i) => o["s" + i] != null));
    root.append(card("Edge-haircut stress test",
      "Expected value if part of your historical average edge does not persist. Same paths, same rules.",
      legend(stressSeries.map((sr) => [sr.name, sr.color, "line"])),
      kChart({ grid: stressGrid, yFmt: compact, yZero: true,
        series: stressSeries.map((sr, i) => ({ name: sr.name, color: sr.color, v: (o) => o["s" + i],
          tip: (o) => money(o["s" + i], true), endLabel: i === 0 ? null : pct(r.stress[i - 1].haircut, 0) })) }),
      table(["Edge haircut", "Best size", `Pass at ${kfmt(rec.k)}`, `EV at ${kfmt(rec.k)}`,
             `Pass at ${kfmt(rec.k_current)}`, `EV at ${kfmt(rec.k_current)}`],
        r.stress.map((sc) => [pct(sc.haircut, 0), kfmt(sc.recommended_k),
          pct(sc.at_base_recommendation.p_pass.p), money(sc.at_base_recommendation.ev, true),
          pct(sc.at_current.p_pass.p), money(sc.at_current.ev, true)]))));

    root.append(card("Every size, in full",
      "All probabilities are over all simulated attempts. Break-even payout = fee ÷ pass probability + activation fee.",
      table(["Size", "Pass", "Pass 95% CI", "Daily loss", "Max loss", "Still running", "Delayed",
             "Median days", "Any payout", "Funded breach", "Net payouts", "EV", "Break-even payout"],
        r.grid.map((g) => [kfmt(g.k), pct(g.p_pass.p), `${pct(g.p_pass.lo)}–${pct(g.p_pass.hi)}`,
          pct(g.p_fail_daily_loss.p), pct(g.p_fail_max_loss.p), pct(g.p_timeout.p),
          pct(g.p_consistency_delay.p), g.days_to_pass_p50 ?? "–",
          `${pct(g.p_payout.p)} (${pct(g.p_payout.lo)}–${pct(g.p_payout.hi)})`,
          pct(g.p_funded_breach.p), money(g.mean_net_payout), money(g.ev, true),
          money(g.breakeven_funded_payout)]),
        (i) => r.grid[i].k === rec.k || r.grid[i].k === rec.k_current)));

    const q = r.inputs.quality, t = r.inputs.trades;
    root.append(card("Inputs and assumptions", null,
      el("p", { text: `${q.n_trades} trades over ${q.n_trading_days} trading days; win rate ${pct(t.win_rate)}, ` +
        `average trade ${money(t.mean)}, worst losing streak ${t.longest_losing_streak}. ` +
        `Intra-trade model: ${cfg.intratrade}. Evaluation cap ${cfg.max_eval_days} days; funded horizon ` +
        `${cfg.funded_days} days${cfg.payout_cap ? `, payout cap ${money(cfg.payout_cap)}` : ""}` +
        `${rs.profit_split ? `, ${pct(rs.profit_split, 0)} profit split` : ""}` +
        `${cfg.fee_refund_on_first_payout ? ", fee refunded with first payout" : ""}.` }),
      r.warnings.length ? el("ul", { class: "warn" }, ...r.warnings.map((w) => el("li", { text: w }))) : null,
      rs.needs_verification.length ? el("h3", { text: "Rule questions still open" }) : null,
      rs.needs_verification.length ? el("ul", {}, ...rs.needs_verification.map((w) => el("li", { text: w }))) : null,
      el("h3", { text: "Rule notes" }), el("ul", {}, ...rs.notes.map((w) => el("li", { text: w }))),
      el("h3", { text: "Sources" }), el("ul", {}, ...rs.source_urls.map((u) => el("li", { text: u })))));

    root.append(el("footer", {}, ...r.disclaimers.map((d) => el("p", { text: d })),
      el("p", { text: `VELOX ${r.velox_version} · schema ${r.schema_version}` })));
  }

  global.renderVelox = renderVelox;
})(typeof window !== "undefined" ? window : globalThis);
