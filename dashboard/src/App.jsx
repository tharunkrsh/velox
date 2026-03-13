import { useState } from "react"
import axios from "axios"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
  AreaChart, Area
} from "recharts"

const API = "http://localhost:8000"

const REGIME_COLORS = {
  bull:     "#00ff88",
  sideways: "#ffcc00",
  bear:     "#ff4444",
}

const DEFAULT_PARAMS = {
  symbols:         "AAPL,MSFT,GOOGL,PEP,CVX",
  start_date:      "2020-01-01",
  end_date:        "2023-12-31",
  initial_capital: 100000,
  strategy:        "ml",
  ml_threshold:    0.6,
  slippage_pct:    0.001,
  commission_pct:  0.001,
}

export default function App() {
  const [params, setParams]   = useState(DEFAULT_PARAMS)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const runBacktest = async () => {
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const payload = {
        ...params,
        symbols: params.symbols.split(",").map(s => s.trim()),
        initial_capital: parseFloat(params.initial_capital),
        ml_threshold:    parseFloat(params.ml_threshold),
        slippage_pct:    parseFloat(params.slippage_pct),
        commission_pct:  parseFloat(params.commission_pct),
      }

      const res = await axios.post(`${API}/backtest`, payload, {
        timeout: 300000,
      })
      setResults(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

const equityData = (() => {
    if (!results?.equity_curve) return []
    let peak = -Infinity
    return results.equity_curve.map((p) => {
      const eq = parseFloat((p.total_equity / (results.equity_curve[0]?.total_equity || 1) * 100).toFixed(2))
      if (eq > peak) peak = eq
      const drawdown = parseFloat(((eq - peak) / peak * 100).toFixed(2))
      return {
        date:     p.timestamp.slice(0, 10),
        equity:   eq,
        buyhold:  p.buyhold_norm,
        drawdown: drawdown,
      }
    })
  })()

  const regimeData = results?.regime_history?.map(r => ({
    date:      r.timestamp.slice(0, 10),
    bull:      r.prob_bull,
    sideways:  r.prob_sideways,
    bear:      r.prob_bear,
    regime:    r.regime,
  })) || []

  return (
    <div style={{
      background: "#0a0a0f",
      minHeight:  "100vh",
      color:      "#e0e0e0",
      fontFamily: "'SF Mono', 'Fira Code', monospace",
      padding:    "32px",
    }}>
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
          `}</style>
          
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 32, fontWeight: 700, letterSpacing: 4, color: "#00ff88", margin: 0 }}>
          VELOX
        </h1>
        <p style={{ color: "#555", fontSize: 12, letterSpacing: 2, margin: "4px 0 0" }}>
          ALGORITHMIC TRADING RESEARCH FRAMEWORK
        </p>
      </div>

      {/* Config panel */}
      <div style={{
        background: "#0f0f1a",
        border:     "1px solid #1e1e2e",
        borderRadius: 8,
        padding:    24,
        marginBottom: 24,
      }}>
        <p style={{ color: "#555", fontSize: 11, letterSpacing: 2, marginBottom: 16 }}>
          BACKTEST CONFIGURATION
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 16 }}>
          {[
            { label: "SYMBOLS",    key: "symbols",         type: "text" },
            { label: "START DATE", key: "start_date",      type: "text" },
            { label: "END DATE",   key: "end_date",        type: "text" },
            { label: "CAPITAL",    key: "initial_capital", type: "number" },
          ].map(({ label, key, type }) => (
            <div key={key}>
              <p style={{ color: "#555", fontSize: 10, letterSpacing: 1, marginBottom: 6 }}>{label}</p>
              <input
                type={type}
                value={params[key]}
                onChange={e => setParams(p => ({ ...p, [key]: e.target.value }))}
                style={{
                  width: "100%", background: "#0a0a0f", border: "1px solid #1e1e2e",
                  borderRadius: 4, padding: "8px 10px", color: "#e0e0e0",
                  fontFamily: "inherit", fontSize: 12, boxSizing: "border-box",
                }}
              />
            </div>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 20 }}>
          <div>
            <p style={{ color: "#555", fontSize: 10, letterSpacing: 1, marginBottom: 6 }}>STRATEGY</p>
            <select
              value={params.strategy}
              onChange={e => setParams(p => ({ ...p, strategy: e.target.value }))}
              style={{
                width: "100%", background: "#0a0a0f", border: "1px solid #1e1e2e",
                borderRadius: 4, padding: "8px 10px", color: "#e0e0e0",
                fontFamily: "inherit", fontSize: 12, boxSizing: "border-box",
              }}
            >
              <option value="ml">LightGBM ML</option>
              <option value="momentum">Momentum</option>
              <option value="pairs">Kalman Pairs</option>
            </select>
          </div>

          {[
            { label: "ML THRESHOLD", key: "ml_threshold" },
            { label: "SLIPPAGE %",   key: "slippage_pct" },
            { label: "COMMISSION %", key: "commission_pct" },
          ].map(({ label, key }) => (
            <div key={key}>
              <p style={{ color: "#555", fontSize: 10, letterSpacing: 1, marginBottom: 6 }}>{label}</p>
              <input
                type="number"
                step="0.001"
                value={params[key]}
                onChange={e => setParams(p => ({ ...p, [key]: e.target.value }))}
                style={{
                  width: "100%", background: "#0a0a0f", border: "1px solid #1e1e2e",
                  borderRadius: 4, padding: "8px 10px", color: "#e0e0e0",
                  fontFamily: "inherit", fontSize: 12, boxSizing: "border-box",
                }}
              />
            </div>
          ))}
        </div>

        <button
          onClick={runBacktest}
          disabled={loading}
          style={{
            background:    loading ? "#1e1e2e" : "#00ff88",
            color:         loading ? "#555" : "#0a0a0f",
            border:        "none",
            borderRadius:  4,
            padding:       "10px 32px",
            fontFamily:    "inherit",
            fontSize:       12,
            fontWeight:    700,
            letterSpacing: 2,
            cursor:        loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "RUNNING BACKTEST..." : "RUN BACKTEST"}
        </button>

        {loading && (
  <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 12 }}>
    <div style={{
      width: 16, height: 16, borderRadius: "50%",
      border: "2px solid #1e1e2e",
      borderTopColor: "#00ff88",
      animation: "spin 0.8s linear infinite",
    }} />
    <p style={{ color: "#555", fontSize: 11, margin: 0 }}>
            {params.strategy === "ml" || params.strategy === "ml+pairs"
        ? "Training ML models and running simulation — this takes ~60 seconds..."
        : "Running simulation — this takes ~20 seconds..."}
    </p>
  </div>
)}

        {error && (
          <p style={{ color: "#ff4444", fontSize: 11, marginTop: 12 }}>
            Error: {error}
          </p>
        )}
      </div>

      {/* Results */}
      {results && (
        <>
          {/* Metrics */}
          <div style={{
            display: "grid", gridTemplateColumns: "repeat(4, 1fr)",
            gap: 12, marginBottom: 24,
          }}>
            {[
              { label: "TOTAL RETURN",   value: `${results.metrics.total_return_pct}%`, color: results.metrics.total_return_pct > 0 ? "#00ff88" : "#ff4444" },
              { label: "SHARPE RATIO",   value: results.metrics.sharpe_ratio,           color: results.metrics.sharpe_ratio > 1 ? "#00ff88" : results.metrics.sharpe_ratio > 0 ? "#ffcc00" : "#ff4444" },
              { label: "MAX DRAWDOWN",   value: `${results.metrics.max_drawdown_pct}%`, color: "#ff4444" },
              { label: "TOTAL TRADES",   value: results.metrics.total_trades,           color: "#e0e0e0" },
              { label: "ANN. RETURN",    value: `${results.metrics.ann_return_pct}%`,   color: "#e0e0e0" },
              { label: "ANN. VOL",       value: `${results.metrics.ann_volatility}%`,   color: "#e0e0e0" },
              { label: "CALMAR RATIO",   value: results.metrics.calmar_ratio,           color: "#e0e0e0" },
              { label: "FINAL EQUITY",   value: `$${results.metrics.final_equity?.toLocaleString()}`, color: "#00ff88" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                background: "#0f0f1a", border: "1px solid #1e1e2e",
                borderRadius: 8, padding: 16,
              }}>
                <p style={{ color: "#555", fontSize: 10, letterSpacing: 1, margin: "0 0 8px" }}>{label}</p>
                <p style={{ color, fontSize: 22, fontWeight: 700, margin: 0 }}>{value}</p>
              </div>
            ))}
          </div>

          {/* Equity curve */}
          <div style={{
            background: "#0f0f1a", border: "1px solid #1e1e2e",
            borderRadius: 8, padding: 24, marginBottom: 16,
          }}>
            <p style={{ color: "#555", fontSize: 11, letterSpacing: 2, marginBottom: 16 }}>
              EQUITY CURVE (NORMALISED TO 100)
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={equityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
                <XAxis dataKey="date" stroke="#333" tick={{ fontSize: 10 }}
                  tickFormatter={d => d.slice(0, 7)}
                  interval={Math.floor(equityData.length / 8)} />
                <YAxis stroke="#333" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: "#0f0f1a", border: "1px solid #1e1e2e", fontSize: 11 }}
                  labelStyle={{ color: "#888" }}
                />
                <Line type="monotone" dataKey="equity"  stroke="#00ff88" dot={false} strokeWidth={2} name="VELOX" />
                <Line type="monotone" dataKey="buyhold" stroke="#4488ff" dot={false} strokeWidth={1.5} strokeDasharray="4 4" name="Buy & Hold" /> 
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Drawdown */}
          <div style={{
            background: "#0f0f1a", border: "1px solid #1e1e2e",
            borderRadius: 8, padding: 24, marginBottom: 16,
          }}>
            <p style={{ color: "#555", fontSize: 11, letterSpacing: 2, marginBottom: 16 }}>
              DRAWDOWN
            </p>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={equityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
                <XAxis dataKey="date" stroke="#333" tick={{ fontSize: 10 }}
                  tickFormatter={d => d.slice(0, 7)}
                  interval={Math.floor(equityData.length / 8)} />
                <YAxis stroke="#333" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: "#0f0f1a", border: "1px solid #1e1e2e", fontSize: 11 }}
                  labelStyle={{ color: "#888" }}
                />
                <Area type="monotone" dataKey="drawdown" stroke="#ff4444"
                  fill="rgba(255,68,68,0.15)" dot={false} name="Drawdown" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Regime probabilities */}
          {regimeData.length > 0 && (
            <div style={{
              background: "#0f0f1a", border: "1px solid #1e1e2e",
              borderRadius: 8, padding: 24, marginBottom: 16,
            }}>
              <p style={{ color: "#555", fontSize: 11, letterSpacing: 2, marginBottom: 16 }}>
                REGIME PROBABILITIES
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={regimeData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
                  <XAxis dataKey="date" stroke="#333" tick={{ fontSize: 10 }}
                    tickFormatter={d => d.slice(0, 7)}
                    interval={Math.floor(regimeData.length / 8)} />
                  <YAxis stroke="#333" tick={{ fontSize: 10 }} domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{ background: "#0f0f1a", border: "1px solid #1e1e2e", fontSize: 11 }}
                  />
                  <Area type="monotone" dataKey="bull"     stackId="1" stroke="#00cc66" fill="rgba(0,204,102,0.3)"    name="Bull" />
                  <Area type="monotone" dataKey="sideways" stackId="1" stroke="#ffcc00" fill="rgba(255,204,0,0.3)"    name="Sideways" />
                  <Area type="monotone" dataKey="bear"     stackId="1" stroke="#ff4444" fill="rgba(255,68,68,0.3)"    name="Bear" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          <p style={{ color: "#333", fontSize: 10, textAlign: "right" }}>
            Completed in {results.duration_secs}s
          </p>
        </>
      )}
    </div>
  )
}