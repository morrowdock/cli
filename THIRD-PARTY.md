# Third-party inventory

Status: verified declared dependencies and versions, 2026-11-11 (installed in a clean venv on macOS, Python 3.14); binary/transitive redistribution review pending. Nothing below is vendored or bundled in this repository.

| Component | Use | Version pinned in CI | License |
| --- | --- | --- | --- |
| cryptography (+ cffi, pycparser) | Runtime: Ed25519 signing of `morrowdock-progress.json` | 50.0.2 (cffi 2.1.1, pycparser 3.12) | Apache-2.0 OR BSD-3-Clause (cffi MIT-0, pycparser BSD-3-Clause) |
| numpy | Runtime: GPT series checks | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pytest, pytest-cov, coverage, iniconfig, pluggy, Pygments, packaging | Optional `[portfolio]` extra: learner portfolio gate | 9.2.0, 7.1.0, 7.17.2, 2.3.1, 1.7.0, 2.21.0, 26.3 | MIT; coverage Apache-2.0; Pygments BSD-2-Clause; packaging Apache-2.0 OR BSD-2-Clause |
| ruff | Optional `[portfolio]` extra | 0.17.4 | MIT |
| mypy (+ mypy_extensions, typing_extensions, librt, ast_serialize, pathspec) | Optional `[portfolio]` extra | 2.4.1 (1.2.0, 4.16.0, 0.16.0, 0.13.1, 1.1.1) | MIT; typing_extensions PSF-2.0; pathspec MPL-2.0 (development only, not distributed: a transitive dependency of mypy in the optional `[portfolio]` extra) |
| torch | Optional `[gpu]` extra and the optional reveal file `torch_path.py`; user-installed | not pinned, not installed in CI | BSD-3-Clause (bundled native libraries need separate review) |
| qrcode[pil] (+ Pillow) | Optional `[qr]` extra; user-installed | not pinned, not installed in CI | BSD; Pillow MIT-CMU |
| Lua 5.4 | External interpreter for the game series and the Lua reference test; user-installed (CI: Ubuntu `lua5.4` package) | distro package | MIT |
| luacheck, StyLua, LuaCov | External tools named by the Lua portfolio gate; user-installed, never installed by the CLI | — | luacheck MIT, LuaCov MIT, StyLua MPL-2.0 (development only, not distributed: a separate program that learners install) |
| actions/checkout, actions/setup-python | CI only, pinned by commit SHA, fetched by CI | — | MIT |
| Contributor Covenant 2.1 | CODE_OF_CONDUCT.md (CC-BY-4.0 snapshot 91532d80ea5f39b6685aff4892bc6854bf2f6d6f); only the enforcement contact is substituted; attribution retained | — | CC-BY-4.0 |

No keys, trust configuration, models, datasets, fonts, images or binary assets are included. Each learner project generates its own Ed25519 key locally in `.morrowdock/signing.key`. No GPL/AGPL/LGPL or custom-license runtime is used. The two MPL-2.0 items are development-only tools that are not distributed; they were accepted as development-only on 2026-11-11.
