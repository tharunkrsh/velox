"""Regenerate the broker-export fixtures (run from the repo root).

MT5 desktop writes UTF-16 LE with a BOM and tab separators; the comma variant is
what a copy/paste or web export produces. Both carry identical trades.
"""

from pathlib import Path

HERE = Path(__file__).parent

MT5_HEADER = ["Time", "Position", "Symbol", "Type", "Volume", "Price", "S / L", "T / P",
              "Time", "Price", "Commission", "Swap", "Profit"]
MT5_ROWS = [
    ["2026.03.02 09:15:00", "5001", "EURUSD", "buy", "1.00", "1.08412", "1.08212", "1.08812",
     "2026.03.02 11:40:00", "1.08610", "-7.00", "0.00", "198.00"],
    # no stop loss / take profit
    ["2026.03.02 14:05:00", "5002", "XAUUSD", "sell", "0.50", "2911.40", "", "",
     "2026.03.02 15:30:00", "2916.80", "-3.50", "0.00", "-270.00"],
    # held across the midnight (CE(S)T) daily-loss rollover
    ["2026.03.02 23:40:00", "5003", "US100.cash", "buy", "2.00", "21010.5", "20960.5", "",
     "2026.03.03 00:20:00", "21035.0", "0.00", "0.00", "49.00"],
    # multi-day hold with swap and a space thousands separator
    ["2026.03.03 10:00:00", "5004", "GBPUSD", "sell", "3.00", "1.27100", "1.27350", "1.26600",
     "2026.03.04 16:45:00", "1.26683", "-21.00", "-9.30", "1 251.00"],
    # totals line at the bottom of the export
    ["", "", "", "", "", "", "", "", "", "", "-31.50", "-9.30", "1 228.00"],
]

TRADOVATE_HEADER = ["symbol", "_priceFormat", "_priceFormatType", "_tickSize", "buyFillId",
                    "sellFillId", "qty", "buyPrice", "sellPrice", "pnl", "boughtTimestamp",
                    "soldTimestamp", "duration"]
TRADOVATE_ROWS = [
    ["MESM6", "-2", "0", "0.25", "111", "112", "2", "5712.25", "5716.50", "$42.50",
     "03/02/2026 08:35:10", "03/02/2026 08:52:40", "17min 30sec"],
    # one short trade filled in two partial fills (sold first)
    ["MESM6", "-2", "0", "0.25", "121", "120", "1", "5720.00", "5713.75", "$(31.25)",
     "03/02/2026 10:05:00", "03/02/2026 09:58:12", "6min 48sec"],
    ["MESM6", "-2", "0", "0.25", "122", "120", "1", "5720.00", "5713.75", "$(31.25)",
     "03/02/2026 10:05:00", "03/02/2026 09:58:12", "6min 48sec"],
    # evening hold across the 17:00 CT session rollover -> next trading day
    ["MNQM6", "-2", "0", "0.25", "131", "132", "3", "20410.00", "20429.75", "$118.50",
     "03/02/2026 16:50:00", "03/02/2026 17:20:30", "30min 30sec"],
    ["MESM6", "-2", "0", "0.25", "141", "142", "2", "5730.00", "5628.75", "$(1,012.50)",
     "03/03/2026 09:10:00", "03/03/2026 09:40:00", "30min"],
]


def _csv(rows, sep):
    def cell(c):
        return f'"{c}"' if (sep == "," and "," in c) else c
    return "\r\n".join(sep.join(cell(c) for c in r) for r in rows) + "\r\n"


if __name__ == "__main__":
    mt5 = [MT5_HEADER, *MT5_ROWS]
    (HERE / "mt5_positions_utf16.tsv").write_bytes(
        b"\xff\xfe" + _csv(mt5, "\t").encode("utf-16-le"))
    (HERE / "mt5_positions_comma.csv").write_text(_csv(mt5, ","), encoding="utf-8")
    (HERE / "tradovate_performance.csv").write_text(
        _csv([TRADOVATE_HEADER, *TRADOVATE_ROWS], ","), encoding="utf-8")
