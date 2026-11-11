# Morrowdock projects: learn, practice, build

Python 3.10+ on macOS or Windows:

```sh
python -m pip install './tools/morrowdock-cli[portfolio]'
mkdir my-gpt
cd my-gpt
morrowdock init gpt
morrowdock learn
morrowdock guide
# Write your own code in src/main.py and your own tests in tests/.
morrowdock check
morrowdock next
```

Series: `gpt` (six released stages), `shell` (four), and first stages with outlines for `agent`, `autopilot`, `game`, and `contributions`. Lua game checks require Lua 5.4 on PATH. GPT uses NumPy; other Python checks use the standard library.

Every stage begins with concepts, a worked example and a short walkthrough. `learn` is read-only and free; `guide` supplies free hints and a neutral “practiced with help” tag. After you try, `morrowdock reveal` exposes a clearly marked worked solution in `.morrowdock/reveals`, preserving your own source. Study it, close it, and rebuild your own version. A reveal never completes the stage. No learning help deducts progress, and retries retain previous successful work. `next` still requires a passing check for your current exact source.

`init` ships src/tests/docs, typed Python interfaces, intentionally unfinished learner-authored test prompts, lint/format/type configs, GitHub Actions CI, a README with demo/architecture Mermaid/decisions/results/lessons, MIT license template, release notes, roadmap, senior-review rubric, demo recipe, blog draft, metric-based resume prompts, stretch goals and a profile README draft. Complete the templates yourself; no achievements, benchmark metrics, CI successes or accepted contributions are invented.

```sh
morrowdock check --portfolio
```

The gate actually runs Ruff lint/format, mypy and pytest with at least 80% combined line/branch coverage. It checks docs and benchmarks, workflow/badges, chosen license, annotated local release tag and notes, three open roadmap items, recorded senior-review evidence, meaningful focused history (at least two commits per released stage) and a clean worktree. Missing tools or evidence produce actionable “NEXT STEP” results. This verifies local readiness, not remote CI or publication. Lua additionally requires luacheck, StyLua and LuaCov (install them for your machine); its gate runs actual Lua tests and reads measured LuaCov coverage. CI workflow scaffolds are provided for your project; they run on GitHub only after you publish your own project repository.

The contributions track teaches license/CONTRIBUTING checks, issue reproduction, regressions, focused forks/branches, issue and PR etiquette, and responding to review. [pytest's contributor queue](https://github.com/pytest-dev/pytest/contribute) and [NumPy's contribution guide](https://numpy.org/devdocs/dev/index.html) are starting points, checked 2026-11-02; verify current availability and maintainer guidance before choosing work. Morrowdock open-core contributions use only an approved public/license scope when available; this release does not assume it is public. Never upload private notes, learner data or credentials. Nothing posts an issue, opens a PR, pushes, creates a Git tag or changes repository history for you.

`check` writes a signed `morrowdock-progress.json`. Import via Files/AirDrop in Projects, or install `[qr]` and use `morrowdock qr`, scan its PNG and paste the text. Signatures protect transfer continuity; this is self-reported local practice. `next` archives each previous starter in `.morrowdock/stage-N`; keep this private directory and its signing key out of Git. Existing write-ups and unrelated files are preserved.

GPT stage 6 uses a small NumPy transformer with a learned readout over fixed transformer features. Its marked optional reveal includes a full-parameter PyTorch training example after your attempt. Install a PyTorch build for your machine separately; CPU is the default and CUDA is explicitly selected. No model, dataset, credentials or paid provider is downloaded/called. Record your own model card, held-out loss and limits. CUDA and native Windows acceptance are pending.

Finished complete series generate a separate `PORTFOLIO-README.md` draft without replacing your writing. Add your own demo and measured results before publishing. `morrowdock-progress.json` is a JSON envelope: base64 `payload`, the project's Ed25519 `publicKey`, and a `signature` over the payload bytes (see `morrowdock_cli/cli.py`).

Developer verification from repo root, after installing `[portfolio]` and Lua:

```sh
python -m unittest discover -s tools/morrowdock-cli/tests -v
python tools/morrowdock-cli/build_catalog.py
```

Catalog generation reproduces the identical packaged CLI catalog from original lessons and references (in the source repository it also writes the app copy). Programs run with normal OS permissions; this local tool is not a hostile-code sandbox.
