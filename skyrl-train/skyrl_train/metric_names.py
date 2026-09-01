"""Metric names shared by training and offline analysis."""

ROLLOUT_FAILURE_FRACTION_METRIC = "generate/failed_trajectory_fraction"
IDENTITY_AWARE_REWARD_METRIC_PREFIX = "generate/reward_shaping/identity_aware"
TOKEN_PROVENANCE_RECONSTRUCTED_FRACTION_METRIC = "generate/token_provenance/reconstructed_fraction"
TIS_ALIGNED_TOKENS_METRIC = "generate/tis/aligned_tokens"
TIS_METRIC_PREFIX = TIS_ALIGNED_TOKENS_METRIC.removesuffix("aligned_tokens")
TIS_EXACT_MATCH_FRACTION_METRIC = "generate/tis/exact_match_fraction"
TIS_LCS_FALLBACK_FRACTION_METRIC = "generate/tis/lcs_fallback_fraction"
TIS_UNALIGNED_FRACTION_METRIC = "generate/tis/unaligned_fraction"
TIS_ALIGNMENT_FAIL_COUNT_METRIC = "generate/tis/alignment_fail_count"
TIS_LCS_FALLBACK_MESSAGES_METRIC = "generate/tis/lcs_fallback_messages"
TIS_LCS_FALLBACK_ALERT_METRIC = "generate/tis/lcs_fallback_alert"
TIS_ALIGNMENT_ALERT_METRIC = "generate/tis/alignment_alert"
