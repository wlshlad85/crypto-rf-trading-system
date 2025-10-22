"""Light tests for analytics.model_comparison."""

from analytics.model_comparison import compare_models, SUPPORTED_MODELS


def test_supported_models():
    assert {"minute_rf", "minute_rf_ensemble", "basic_rf"}.issubset(SUPPORTED_MODELS)


def test_compare_models_runs_smoke():
    # Run a small comparison to ensure it executes without errors.
    results = compare_models(["basic_rf"], minutes=600, horizon=5)
    assert "basic_rf" in results
    res = results["basic_rf"]
    # Expect summary to contain core metrics keys (may be zero on small sims)
    assert isinstance(res.summary, dict)
    for key in ["total_return", "annualized_return", "sharpe_ratio", "win_rate"]:
        assert key in res.summary

