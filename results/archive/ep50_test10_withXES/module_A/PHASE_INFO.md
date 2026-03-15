# Archive: ep50_test10_withXES — Module A

**Archived on**: 2026-03-15 06:52 UTC

**Epochs**: 50

**Test ratio**: 0.1 (10%)

**Tickers**: ['SPY', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'GDX', 'IWM', 'XES']

**Phase**: 2a — initial run with XES, lower epochs

**Files**: ['dlinear_meta_20260315.json', 'dlinear_ret_meta_20260315.json', 'crossformer_ret_meta_20260315.json', 'eval_results_20260315.json', 'crossformer_meta_20260315.json', 'dlinear_prc_meta_20260315.json', 'crossformer_prc_meta_20260315.json', 'performance_history.json']

## Key Results

See eval_results_*.json for full backtest metrics.

### Notes
- XES included in equity universe
- First run proving RET > PRC for ETFs
- Buy & Hold alignment bug present (fixed in Phase 2b)
