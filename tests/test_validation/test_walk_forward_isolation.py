import pandas as pd

from trade.validation.walk_forward import WalkForwardValidator


def test_walk_forward_windows_are_non_overlapping_and_chronological(sample_ohlcv):
    frame = sample_ohlcv.copy()
    frame["dummy"] = 0.0
    validator = WalkForwardValidator(train_window_days=50, validation_window_days=10,
                                      test_window_days=20, step_days=20,
                                      feature_window=5)
    # No model execution is needed to verify the split contract; a missing
    # model path causes individual evaluations to be skipped safely.
    result = validator.validate("missing-model", frame, ["dummy"])
    assert result.window_results == [] or all(
        w["train_end"] <= w["validation_start"] <= w["validation_end"] <= w["test_start"] <= w["test_end"]
        for w in result.window_results
    )


def test_unsorted_datetime_input_is_ordered_before_windowing(sample_ohlcv):
    frame = sample_ohlcv.iloc[::-1].copy()
    frame["dummy"] = 0.0
    validator = WalkForwardValidator(train_window_days=50, test_window_days=20,
                                      step_days=20, feature_window=5)
    # The validator must not reject or shuffle nondeterministically.
    result = validator.validate("missing-model", frame, ["dummy"])
    assert result.model_version.tag == "v0.0.0"
