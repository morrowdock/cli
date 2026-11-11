# morrowdock CLI

Early preview — no stable release or compatibility promise yet. Learner-owned "build something" projects: guided GPT and shell series, plus first stages of agent, autopilot, game and contributions series. Code: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Names, mascot and logos are not licensed ([TRADEMARKS.md](TRADEMARKS.md)). Contributions need a DCO sign-off ([CONTRIBUTING.md](CONTRIBUTING.md)).

The package lives in [`tools/morrowdock-cli/`](tools/morrowdock-cli/README.md), the same path as in its source repository, so its tests and catalog generator run unchanged. Quick start (Python 3.10+):

```sh
python -m pip install './tools/morrowdock-cli[portfolio]'
mkdir my-gpt && cd my-gpt
morrowdock init gpt
morrowdock learn
```

Developer checks from the repository root (Lua 5.4 as `lua` on PATH for the Lua reference test):

```sh
PYTHONPATH=tools/morrowdock-cli/tests python -m unittest discover -s tools/morrowdock-cli/tests -v
python tools/morrowdock-cli/build_catalog.py && git diff --exit-code -- tools/morrowdock-cli/morrowdock_cli/catalog.json
```

Linux CI runs both on pushes and pull requests with pinned dependencies (`tools/morrowdock-cli/ci-constraints.txt`). Windows cp1252 and the native app decoder are checked in the private app repository, which also holds the app copy of the catalog; the parity checks that compare against it skip here (three tests skip; two more comparisons run only when the app copy exists). Programs run with normal OS permissions; this local tool is not a hostile-code sandbox.

Source and dependencies: [EXTRACTION.json](EXTRACTION.json), [THIRD-PARTY.md](THIRD-PARTY.md).

Public contact, Code of Conduct concerns, and privacy/deletion requests: [hello@morrowdock.app](mailto:hello@morrowdock.app). For vulnerability reporting, see [SECURITY.md](SECURITY.md).
