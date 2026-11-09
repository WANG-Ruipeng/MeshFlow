# Review package

Open ../REPORT.md and ../gallery/index_portable.html. All experiment CSV bytes, 24 JPEG views, actual old/new source code, diffs, registrations, receipts and ledgers are included. No adverse metric or request is removed. Large raw pair JSON, geometry arrays, weights and original PNGs remain local and are individually indexed with SHA256 in omitted_local_assets.json. gallery/index.html is the local full-PNG viewer; use index_portable.html inside this ZIP.

Compression is standard ZIP DEFLATE. The source snapshot preserves existing source bytes; code.diff consists only of the new experiment files. old_source.diff is empty after registered original source hashes pass. completion.json and archive verification are produced outside the ZIP after packaging to avoid a circular ZIP self-hash.
