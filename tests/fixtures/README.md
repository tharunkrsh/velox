# Test fixtures

`topstep_sample.csv`: naive America/Chicago timestamps. Hand computation for
`velox replay --preset topstep-50k --tz America/Chicago` (17:00 CT rollover):

| Trading day | Trades | Balance | Notes |
|---|---|---|---|
| Mon 01-08 | +600, +700 | 51,300 | |
| Tue 01-09 | +900 | 52,200 | |
| Wed 01-10 | -200 | 52,000 | |
| Thu 01-11 | +1,200 (closed Wed 18:45 CT, so it counts for Thu) | 53,200 | profit 3,200 >= 3,000; best day 1,300 / 3,200 = 41% <= 55% -> PASS (attempt 1, day 4) |
| Fri 01-12 | -900 | 49,100 | attempt 2 on a fresh 50,000; MLL floor 48,000 |
| Mon 01-15 | -900 | 48,200 | |
| Tue 01-16 | -300 | 47,900 | <= 48,000 -> FAIL eod_trailing_max_loss |

`mt5_positions.csv`: synthetic MT5 Positions-tab layout (not a real export).
Trade 1002 opens before 1001 closes (one overlap).
