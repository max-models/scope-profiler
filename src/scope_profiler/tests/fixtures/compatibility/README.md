# Compatibility fixtures

These files are immutable examples of every supported on-disk profile format.
They are hand-built against the documented layouts and deliberately not
produced by the writer during the test run: the reader must remain compatible
with fixed historical layouts even after the current writer changes.

HDF5 files are gzip-compressed and base64-encoded so they can be reviewed and
updated through normal text patches. Native traces are base64-encoded. Decode
them only into pytest's temporary directory; do not replace a fixture merely
because a new writer emits different bytes. Add a new fixture when a format
version changes.

All fixtures describe rank 0 and one region named `solve`, with starts
`[10, 20, 40]`, ends `[15, 33, 55]`, source line 7 where the format supports
it, and the tag `golden` where supported.
