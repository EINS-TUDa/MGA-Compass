# mga-format

The data format in which [mga-surveyor](../mga-surveyor) hands a computed near-optimal set to
[mga-compass](../mga-compass) (the web platform and the Python API): a directory of Parquet
files and a `manifest.json`. The format is specified in
[`docs/reference/near-optimal-set-format.md`](../docs/reference/near-optimal-set-format.md).

This package reads, validates and writes it. It depends only on pandas, pyarrow and
pydantic, so both packages, and other tools, can use it without depending on each other.
