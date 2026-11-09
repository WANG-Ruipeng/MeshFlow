# Confirmation visual timestamp compatibility

The first root freeze attempt rejected Windows seven-digit fractional UTC timestamps under Python 3.10 datetime.fromisoformat. No freeze receipt was written on that rejected attempt.

All 16 original root request documents are preserved in root_before/. Only the 32 image viewed_at_utc strings were truncated from 100 ns precision to microsecond precision; each full original is retained as viewed_at_utc_original in the final note. Observation text, uncertainty, image identity/hash and actual-view declarations were unchanged. root_receipt.json enumerates every normalization. The frozen merger source was not changed.

The next freeze completed with 16 requests, 32 output observations and 32 original PNGs; SHA256 of the resulting receipt is 59c618fc6a79baea926a401d04cd4317444c17e038954260b73ba10734addb8e. This was a presentation metadata compatibility correction, not a training/sampling retry, numerical failure or additional observation.
