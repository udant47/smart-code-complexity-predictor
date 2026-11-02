# Security review (Phase 18)

SmartCode analyses untrusted Python source. This document records what was
reviewed, what was found and changed, and what risks remain. Every claim
below is backed by an automated test, which is named next to it.

## 1. Submitted code is never executed

Submitted code is only read by `ast.parse` and `tokenize`. It is never
imported, executed, evaluated or compiled to bytecode.

| Check | Result | Test |
|---|---|---|
| Static scan of `app.py` and every module in `src/` for `exec`, `eval`, `compile`, `__import__`, `breakpoint`, `input`, `os.system`/`popen`/`spawn*`/`exec*`/`fork`, `importlib.import_module`, any `shell=` argument, and imports of `subprocess`, `pty`, `runpy`, `importlib`, `ctypes`, `marshal`, `pickle`, `shelve`, `code`, `multiprocessing`, `socket` | none found | `test_security.py::test_no_execution_primitives_shell_or_dynamic_imports` |
| Runtime trap while analysing 15 hostile programs: `exec`, `eval`, `os.system`, `os.popen`, `os._exit`, `os.kill`, `subprocess.Popen` and `subprocess.run` fail the test if called; `compile` is allowed only in AST-only mode (that is what `ast.parse` uses); dynamic imports and file writes are recorded | none happened | `test_security.py::test_hostile_code_is_never_executed` |

The hostile programs would visibly misbehave if run. They would:

- kill the test process with `os._exit` or `SIGKILL`;
- loop forever;
- raise `SystemExit`;
- shell out, write files, or call `exec`/`eval`;
- import modules that don't exist;
- run code from decorators, class bodies, default arguments, annotations or `if __name__ == "__main__"`;
- recurse without end, or try to allocate 8 TB of memory.

All 15 are analysed normally. No marker file appears, and no `evil` module is imported. The same holds for the
Streamlit service layer and for the bytes/upload path (`parse_bytes`).

## 2. Model files (pickles) are verified before loading

`joblib` files are pickles, and loading a pickle can run arbitrary code.

**Finding:** before Phase 18, `predictor.load_model` called `joblib.load` on whatever file sat at the model path.

**Change:** `src/model_integrity.py` now runs `check_file` before *every* unpickling call. It requires:

1. a regular file (symbolic links are refused);
2. no larger than 512 MB;
3. a SHA-256 equal to the hash recorded in `model_metadata.json` under `integrity` when the model was trained.

| Check | Test |
|---|---|
| `joblib.load` appears only in `predictor.load_model`, after `check_file` | `test_security.py::test_unpickling_only_happens_in_the_guarded_model_loader` |
| A malicious pickle (it creates a marker file when unpickled) placed as the model, with and without a recorded hash, is refused and never unpickled | `test_model_integrity.py::test_malicious_pickle_is_never_unpickled` |
| Missing hash → `model_unverified`; tampered file → `model_integrity_failed`; symlink and oversize are refused; the preprocessing file is checked too | `test_model_integrity.py` |
| `train_model` and `evaluate_model` record hashes after saving | `test_model_integrity.py::test_training_commands_record_hashes` |

Commands:

- `python -m src.model_integrity verify` checks the installed files.
- `python -m src.model_integrity register` records hashes for a model you trained yourself before Phase 18.

**Limitation:** the hash detects a replaced, swapped or corrupted file. It is not authentication. Someone who can
already edit both the model file and its metadata on your computer can bypass it. Model files are never accepted
from users, and the app has no upload widget (`test_no_upload_widget_and_safe_streamlit_config`).

## 3. File, network and shell operations

Modules used during analysis write no files. This is checked statically by `test_runtime_modules_do_not_write_files`,
and at runtime by the hostile-code tests, which record every `open(..., "w"/"a"/"x"/"+")`.

The only file writers are command-line tools run by the developer, and they write fixed paths inside the project:

- `train_model`
- `tune_model`
- `evaluate_model`
- `eda`
- `dataset_loader`
- `preprocessing`
- `explainability.write_outputs`
- `model_integrity register`

`dataset_loader` downloads from pinned HTTPS URLs and verifies SHA-256 checksums. Its zip extraction skips hidden
paths and `..`, and checks that every resolved path stays inside the target folder (protection against "zip slip").

## 4. Input limits and robustness

- **Size:** 200 KB and 5,000 lines, enforced before parsing. Exact-boundary tests cover lines, bytes and multi-byte
  characters, and confirm that `ast.parse` is not called on oversized input.
- **Nesting:** parser `RecursionError`/`MemoryError` are reported as `too_complex`. Analysis passes are iterative.
  Deep but valid trees just below the parser's limit (unary/attribute chains, 2,000-arm `elif` chains, 90 nested
  loops) were probed and are tested.
- **Finding, fixed:** a lone Unicode surrogate (`'\udcff'`) crashed the parser with `UnicodeEncodeError`. It is now
  reported as an `encoding` error with the line number. The UI service skips the cache for such text, because the
  cache cannot hash it.
- **Finding, fixed:** files near the limit with hundreds of functions took 27–35 s. There were two quadratic steps:
  `ast.get_source_segment` re-split the whole file for every function, and the forest was called once per function.
  Lines are now split once (identical output, verified on 3,737 real functions), and per-function ML predictions
  run in one batch (identical results, tested). The worst case measured is now about 5 s.
- The UI service turns any internal exception (including `RecursionError` and `MemoryError`) into a short message
  with a reference id. Tracebacks are logged to the terminal, never shown (`showErrorDetails = false`).
- All user-derived text is HTML-escaped before it is placed in custom HTML (`test_frontend_widgets_escape_user_text`).

## 5. Correctness finding (honesty of estimates)

A call into code outside the file that received a whole input collection (`np.linalg.inv(a)`,
`library.solve(xs)`) was assumed to be O(1), with low confidence. Its static estimate now says **Unknown** instead.
Calls on single items (`f(x)` inside `for x in xs`) keep the documented O(1) assumption.

## 6. Network exposure

`.streamlit/config.toml` sets `server.address = "localhost"`, so the app is reachable only from this computer. The
app has no login. To share it on a network or the internet, put it behind authentication and rate limiting: one
analysis at the size limit uses a CPU core for a few seconds.

## 7. Remaining risks

- The integrity check is not a signature (see section 2).
- Deliberately crafted inputs within the limits can still take several seconds each (CPU cost).
- Dependencies (Streamlit, scikit-learn, pandas, numpy, plotly) are trusted as installed. Keep them updated.
