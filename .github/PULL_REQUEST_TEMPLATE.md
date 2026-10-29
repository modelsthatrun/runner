<!-- For a new or changed model adapter. For tool changes, describe the change and how you tested it. -->

**Model:** <!-- name, models/<id>, sizes -->

**Licenses, from primary sources** (URL and the date you read it):
- Code: <!-- the repository's LICENSE file -->
- Weights, per size: <!-- each checkpoint's model card or release notes -->

**Checks.** The `mtr-result` line of `./mtr check <id> --size <size>` for each size:
```json
```

**Accuracy (optional).** `./mtr eval` against what the paper reports, with its exact source ("paper, table 2"):

- [ ] `adapter.py` follows the official README: default input size, loading code, inference call
- [ ] `pyproject.toml` pins the official package and torch, and `uv.lock` is committed and current
- [ ] No records, weights or outputs are in the diff
- [ ] Any patch is in `fixes/` with a README that says why
