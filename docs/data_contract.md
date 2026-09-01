# Market-data contract

`OBSERVATION_FEATURES` is the sole allow-list for policy, live inference,
regime detection, and risk-model inputs. `TARGET_COLUMNS` holds supervised
labels; `METRICS_ONLY_COLUMNS` holds reporting values. Neither category may be
passed to an observation.

Data is chronologically ordered before training splits: training uses the
earliest segment, while validation/evaluation consumes later rows. Any scaler
must be fitted with the training segment only; validation and test data must
only call its `transform` method. Indicators are rolling/past-only and their
initial missing values are set to zero rather than back-filled from future
bars.
