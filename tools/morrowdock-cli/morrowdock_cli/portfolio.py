"""Portfolio evidence checks. Never publishes, opens issues, or writes Git history."""
import json
import math
from pathlib import Path
import re
import subprocess
import sys

SENIOR_RUBRIC = [
    'Code quality: trace one end-to-end path; identify coupling, invariants and error handling.',
    'Tests: show an intentional regression caught by a test; review edge cases and coverage gaps.',
    'Documentation: follow quickstart from a clean checkout; compare demo claims to measured results.',
    'Naming: check that public names communicate units, shapes, ownership and side effects.',
    'Security: review inputs, subprocess boundaries, secrets, dependency licenses and private data.',
]

def readme(series):
    return f'''# {series['title']}

TODO: Write a one-line pitch describing the actual artifact.

![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-OWNER_MEASUREMENT-blue)

## Demo
TODO: Link a GIF/video or playable demo and show one representative result.

## Quickstart
TODO: Document environment, install, commands, expected output, and a clean-checkout test.

## Architecture
```mermaid
flowchart LR
    Input --> Core
    Core --> Result
```
TODO: Explain the real boundaries and replace the sample diagram as needed.

## Design decisions
TODO: Explain two tradeoffs and one rejected alternative with evidence.

## Results
TODO: Report measured quality/runtime with units, baseline, seed, device and reproduction command.

## Lessons
TODO: Describe a bug, the observation that exposed it, and what changed in your reasoning.

## Next steps
TODO: Choose one stretch goal from docs/STRETCH.md and link your roadmap.

## Learning notes
TODO: Describe help used, including optional reveals, neutrally. No deductions for learning.

## Resume bullets
TODO: Replace metric placeholders with your own measured results from docs/benchmarks.json.
'''

def starter_files(series):
    lua = series['id'] == 'game'
    files = {
        'README.md': readme(series),
        '.gitignore': '.morrowdock/\n__pycache__/\n.venv/\n.pytest_cache/\n.mypy_cache/\n.ruff_cache/\n.coverage\ncoverage.json\nluacov.*\nmorrowdock-progress.json\nmorrowdock-progress.png\n',
        'LICENSE': 'MIT License\n\nCopyright (c) TODO_YEAR TODO_OWNER\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n',
        'CHANGELOG.md': '# Release notes\n\n## Unreleased\nTODO: Name a release, its behavior, verification and limitations. Tag it locally when ready.\n',
        'docs/ROADMAP.md': '# Roadmap\n\n- [ ] TODO: Define a correctness improvement and acceptance check.\n- [ ] TODO: Define a performance experiment and measurement.\n- [ ] TODO: Define a usability improvement and demo.\n',
        'docs/REVIEW.md': '# Senior engineer release review\n\nWrite your own evidence and findings; Marty reviews this rubric, not authors your solution.\n\n' + '\n\n'.join('- [ ] '+line+'\nTODO: Evidence, finding, and resolution.' for line in SENIOR_RUBRIC)+'\n',
        'docs/BLOG-DRAFT.md': '# Technical post draft\n\n## Problem\nTODO: Who needs this, constraints, and a small motivating example.\n\n## Approach\nTODO: Walk through your architecture and one design decision.\n\n## Results\nTODO: Reproducible measurements, baseline, graph/table, and limits.\n\n## Lessons\nTODO: A mistaken assumption, a debugging story, and the next experiment.\n',
        'docs/benchmarks.json': json.dumps({'measurements': []}, indent=2)+'\n',
        'docs/DEMO.md': series['demoRecipe']+'\n',
        'docs/STRETCH.md': '# Stretch goals\n\n'+'\n'.join('- '+x for x in series['stretchGoals'])+'\n',
        'docs/RESUME.md': '# Evidence-backed resume drafts\n\n'+'\n'.join('- '+x for x in series['resumeBullets'])+'\n\nReplace [measured ...] with results you can reproduce; omit unsupported claims.\n',
        'docs/PROFILE-README.md': '# Profile README draft\n\nTODO: Your interests, what you built yourself, and what you are exploring next.\n\n## Suggested pins\nChoose your strongest finished GPT, shell, autopilot or game repo; add one real contribution after review. Use your own links and explain your role. No repo is pinned automatically.\n',
    }
    if lua:
        files.update({
            '.luacheckrc': 'std = "lua54"\nmax_line_length = 100\nglobals = {"love", "step"}\n',
            'stylua.toml': 'column_width = 100\nindent_type = "Spaces"\nindent_width = 4\n',
            '.luacov': 'return { include = { "src/.*" }, statsfile = ".morrowdock/luacov.stats.out", reportfile = ".morrowdock/luacov.report.out" }\n',
            'tests/test_project.lua': '-- Write your own boundary and regression tests.\ndofile("src/main.lua")\nerror("TODO: replace this reminder with tests you wrote")\n',
            '.github/workflows/ci.yml': '''name: CI
on: [push, pull_request]
permissions:
  contents: read
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y lua5.4 luarocks cargo
      - run: sudo luarocks --lua-version=5.4 install luacheck && sudo luarocks --lua-version=5.4 install luacov
      - run: cargo install stylua --locked
      - run: mkdir -p .morrowdock && luacheck src tests && stylua --check src tests
      - run: lua -lluacov tests/test_project.lua && luacov
''',
        })
    else:
        files.update({
            'pyproject.toml': '''[tool.ruff]
line-length = 100
target-version = "py310"
[tool.ruff.lint]
select = ["E", "F", "I"]
[tool.mypy]
python_version = "3.10"
disallow_untyped_defs = true
ignore_missing_imports = true
[tool.pytest.ini_options]
pythonpath = ["src"]
[tool.coverage.run]
source = ["src"]
branch = true
[tool.coverage.report]
fail_under = 80
''',
            'requirements-dev.txt': 'numpy>=1.24\npytest>=8\npytest-cov>=5\nruff>=0.6\nmypy>=1.10\n',
            'tests/test_project.py': '"""Write behavioral tests and at least one regression test for your own code."""\n\ndef test_write_your_own_regression() -> None:\n    raise AssertionError("Replace this reminder with your own meaningful tests")\n',
            '.github/workflows/ci.yml': '''name: CI
on: [push, pull_request]
permissions:
  contents: read
jobs:
  verify:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: python -m pip install -r requirements-dev.txt
      - run: python -m ruff check src tests
      - run: python -m ruff format --check src tests
      - run: python -m mypy src
      - run: python -m pytest --cov=src --cov-branch --cov-fail-under=80
''',
        })
    return files


def check(root, series, state):
    """Actual local tools must pass; docs/tag checks do not claim remote CI success."""
    results=[]
    def add(name, ok): results.append((name, bool(ok)))
    def text(name):
        p=root/name
        return p.read_text(encoding='utf-8') if p.is_file() and p.stat().st_size<1_000_000 else ''
    def command(args):
        try: return subprocess.run(args,cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120).returncode==0
        except (OSError,subprocess.TimeoutExpired): return False
    add('all released stages completed',series['complete'] and len(state['completed'])==len(series['stages']))
    add('src / tests / docs structure', all((root/p).is_dir() and any((root/p).iterdir()) for p in ['src','tests','docs']))
    lua=series['id']=='game'
    commands=[['luacheck','src','tests'],['stylua','--check','src','tests']] if lua else [[sys.executable,'-m','ruff','check','src','tests'],[sys.executable,'-m','ruff','format','--check','src','tests'],[sys.executable,'-m','mypy','src']]
    add('lint and format configured', bool(text('.luacheckrc') and text('stylua.toml')) if lua else all(x in text('pyproject.toml') for x in ['[tool.ruff]','[tool.mypy]','disallow_untyped_defs = true']))
    for args in commands: add(' '.join(args[2:] if args[0]==sys.executable else args), command(args))
    if lua:
        add('tests with measured coverage', command(['lua','-lluacov','tests/test_project.lua']) and command(['luacov']))
        report=text('.morrowdock/luacov.report.out'); totals=re.findall(r'Total\s+\d+\s+\d+\s+([\d.]+)%?',report)
        add('coverage at least 80%', bool(totals) and float(totals[-1])>=80)
    else:
        add('tests with at least 80% line and branch coverage', command([sys.executable,'-m','pytest','--cov=src','--cov-branch','--cov-fail-under=80','--cov-report=json:.morrowdock/coverage.json']))
    workflow=text('.github/workflows/ci.yml')
    add('CI runs lint and tests', ('ruff' in workflow and 'pytest' in workflow and 'mypy' in workflow) if not lua else all(x in workflow for x in ['luacheck','stylua','luacov']))
    readme_text=text('README.md')
    sections=['Demo','Quickstart','Architecture','Design decisions','Results','Lessons','Next steps','Learning notes']
    add('README completed with evidence', not re.search(r'TODO|OWNER|REPO|\[measured',readme_text) and all(len(re.search(r'(?im)^## '+re.escape(s)+r'\s*\n([\s\S]*?)(?=\n## |\Z)',readme_text).group(1).strip())>=25 if re.search(r'(?im)^## '+re.escape(s)+r'\s*\n([\s\S]*?)(?=\n## |\Z)',readme_text) else False for s in sections))
    add('architecture Mermaid', '```mermaid' in readme_text and '-->' in readme_text)
    add('CI and coverage badges', '/actions/workflows/ci.yml/badge.svg' in readme_text and 'coverage' in readme_text.lower() and '![' in readme_text)
    demo=re.search(r'(?im)^## Demo\s*\n([\s\S]*?)(?=\n## |\Z)',readme_text)
    links=re.findall(r'https://[^\s)]+|[\w/.-]+\.(?:gif|mp4|webm|html)',demo.group(1)) if demo else []
    add('linked demo artifact', any(link.startswith('https://') or (root/link).is_file() for link in links))
    license=text('LICENSE'); add('chosen MIT or Apache license', ('MIT License' in license or 'Apache License' in license) and len(license)>500 and 'TODO' not in license)
    notes=text('CHANGELOG.md'); add('release notes written', len(notes)>120 and 'TODO' not in notes and '## v' in notes)
    roadmap=text('docs/ROADMAP.md'); add('at least three open roadmap items', roadmap.count('- [ ]')>=3 and 'TODO' not in roadmap)
    review=text('docs/REVIEW.md'); add('senior review evidence recorded', review.count('- [x]')>=5 and 'TODO' not in review and all(word.lower() in review.lower() for word in ['quality','tests','documentation','naming','security']))
    try:
        measures=json.loads(text('docs/benchmarks.json'))['measurements']
        valid=bool(measures) and all(isinstance(m.get('value'),(int,float)) and math.isfinite(m['value']) and not isinstance(m.get('value'),bool) and all(isinstance(m.get(k),str) and len(m[k].strip())>1 for k in ['metric','unit','command','environment','baseline']) for m in measures)
    except (ValueError,KeyError,TypeError): valid=False
    add('reproducible benchmark measurements',valid)
    for path in ['docs/BLOG-DRAFT.md','docs/RESUME.md','docs/DEMO.md']:
        value=text(path); add(path+' completed',len(value)>100 and 'TODO' not in value and '[measured' not in value)
    help_used=state.get('practicedWithHelp',[])
    add('learning help disclosed neutrally', not help_used or all(s in readme_text for s in help_used))
    try:
        git=lambda *args: subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL,timeout=10).strip()
        messages=git('log','--format=%s','-30').splitlines()
        add('focused meaningful commit history',len(messages)>=len(series['stages'])*2 and all(len(m)>=12 and not re.match(r'(?i)^(wip|fix|update|stuff|temp)$',m) for m in messages))
        add('clean tracked worktree',not git('status','--porcelain'))
        tags=git('tag','--points-at','HEAD').splitlines()
        add('annotated release tag at HEAD', any(git('cat-file','-t',t)=='tag' for t in tags))
    except (OSError,subprocess.SubprocessError):
        add('focused meaningful commit history',False); add('clean tracked worktree',False); add('annotated release tag at HEAD',False)
    return results
