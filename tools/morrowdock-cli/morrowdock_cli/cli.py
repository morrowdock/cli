"""Cross-platform local practice. Learner programs run with the owner's OS permissions."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from importlib.resources import files
from . import portfolio as portfolio_tools
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption


def catalog():
    return json.loads(files('morrowdock_cli').joinpath('catalog.json').read_text(encoding='utf-8'))['series']


def atomic(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        name = f.name
        f.write(data)
    try:
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def write_json(path, obj):
    atomic(path, json.dumps(obj, indent=2, ensure_ascii=False).encode('utf-8'))


def digest(root, stage):
    h = hashlib.sha256()
    for name in sorted(stage['starterFiles']):
        h.update(name.encode()); h.update(b'\0'); h.update((root / name).read_bytes()); h.update(b'\0')
    return h.hexdigest()


def envelope(payload, key):
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    b64 = lambda b: base64.b64encode(b).decode('ascii')
    return dict(payload=b64(raw), publicKey=b64(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)), signature=b64(key.sign(raw)))


def verify(doc):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    raw = base64.b64decode(doc['payload'], validate=True)
    Ed25519PublicKey.from_public_bytes(base64.b64decode(doc['publicKey'], validate=True)).verify(base64.b64decode(doc['signature'], validate=True), raw)
    return json.loads(raw)


def scaffold(root, stage):
    # Preflight all paths before writing any; never replace learner work.
    for name in stage['starterFiles']:
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or (root / name).exists():
            raise ValueError(f'Keep your existing {name}; choose an empty project folder.')
    for name, source in stage['starterFiles'].items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source, encoding='utf-8')


def initialize(series_id, root):
    series = next((s for s in catalog() if s['id'] == series_id), None)
    if series is None: raise ValueError('Choose: ' + ', '.join(s['id'] for s in catalog()))
    root.mkdir(parents=True, exist_ok=True)
    if (root / '.morrowdock').exists(): raise ValueError('This project is already initialized.')
    scaffold(root, dict(starterFiles={**series['stages'][0]['starterFiles'], **series.get('supplementalFiles', {}), **portfolio_tools.starter_files(series)}))
    state = root / '.morrowdock'; state.mkdir()
    key = Ed25519PrivateKey.generate()
    keypath = state / 'signing.key'
    keypath.write_bytes(key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()))
    if os.name != 'nt': keypath.chmod(0o600)
    write_json(state / 'state.json', dict(series=series_id, stage=0, projectID=str(uuid.uuid4()), completed=[], revision=0, attempted=[], practicedWithHelp=[]))

    print(f"Ready: {series['title']}. Open src/main.{ 'lua' if series_id == 'game' else 'py' }, start with morrowdock learn, then morrowdock guide. Check your own version when ready.")


def load(root):
    state = json.loads((root / '.morrowdock/state.json').read_text())
    series = next(s for s in catalog() if s['id'] == state['series'])
    return state, series, series['stages'][state['stage']]


def run_tests(root, stage):
    # Tests are bundled, not read from an editable test file in the learner repo.
    # Hidden means held back from the lesson, not secret from the local machine owner.
    source = '\n'.join((root / n).read_text(encoding='utf-8') for n in stage['starterFiles'])
    results = []
    for test in stage['visibleTests'] + stage['hiddenTests']:
        suffix = '.lua' if stage['language'] == 'lua' else '.py'
        with tempfile.TemporaryDirectory(prefix='morrowdock-check-') as folder:
            script = Path(folder) / ('check' + suffix)
            script.write_text(source + '\n' + test['source'], encoding='utf-8')
            command = ['lua', str(script)] if suffix == '.lua' else [sys.executable, '-I', str(script)]
            try:
                # Discard program output to keep runaway printing from filling memory/disk.
                proc = subprocess.run(command, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                results.append((test['name'], proc.returncode == 0))
            except subprocess.TimeoutExpired:
                results.append((test['name'] + ' (time limit)', False))
            except FileNotFoundError:
                raise ValueError('Install Lua 5.4 and add lua to PATH, then try again.')
    return results


def save_progress(root, state, series):
    key = Ed25519PrivateKey.from_private_bytes((root / '.morrowdock/signing.key').read_bytes())
    state['revision'] += 1
    payload = dict(schema=1, catalogVersion=1, projectID=state['projectID'], seriesID=series['id'], revision=state['revision'], completed=state['completed'], practicedWithHelp=state.get('practicedWithHelp', []), issuedAt=datetime.now(timezone.utc).isoformat())
    # Persist state first; exports can safely be regenerated on the next check.
    write_json(root / '.morrowdock/state.json', state)
    write_json(root / 'morrowdock-progress.json', envelope(payload, key))


def check(root, portfolio_ready=False):
    state, series, stage = load(root)
    before = digest(root, stage)
    results = run_tests(root, stage)
    for name, ok in results: print(('PASS' if ok else 'TRY AGAIN') + ' · ' + name)
    passed = bool(results) and all(ok for _, ok in results) and before == digest(root, stage)
    state.setdefault('attempted', [])
    if stage['id'] not in state['attempted']: state['attempted'].append(stage['id'])
    if passed:
        state['completed'] = [c for c in state['completed'] if c['stageID'] != stage['id']]
        state['completed'].append(dict(stageID=stage['id'], sourceSHA256=before, testsPassed=len(results)))
        print('All checks passed. Explain your approach, then run morrowdock next.')
    else:
        print('Next hint: ' + stage['hints'][0])
    save_progress(root, state, series)
    if passed and len(state['completed']) == len(series['stages']) and series['complete']:
        portfolio(root, series)
    if portfolio_ready:
        checks = portfolio_tools.check(root, series, state)
        for name, ok in checks: print(('READY' if ok else 'NEXT STEP') + ' · ' + name)
        print('Portfolio checks describe local evidence. The owner reviews and publishes.')
        passed = passed and all(ok for _,ok in checks)
    return 0 if passed else 1


def portfolio(root, series):
    target = root / 'PORTFOLIO-README.md'
    if target.exists(): return  # preserve owner write-up
    target.write_text(portfolio_tools.readme(series), encoding='utf-8')


def advance(root):
    state, series, stage = load(root)
    record = next((c for c in state['completed'] if c['stageID'] == stage['id']), None)
    if not record or record['sourceSHA256'] != digest(root, stage):
        raise ValueError('Run morrowdock check on your current work before advancing.')
    if state['stage'] + 1 == len(series['stages']):
        if series['complete']: portfolio(root, series); print('Series complete! Review PORTFOLIO-README.md and add your demo.')
        else: print('Starter complete. The next stages are outlined in the app and are not released yet.')
        return
    archive = root / '.morrowdock' / ('stage-' + str(state['stage'] + 1))
    if archive.exists(): raise ValueError('A stage archive already exists; keep it and resolve before advancing.')
    archive.mkdir()
    for name in stage['starterFiles']:
        p = archive / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((root / name).read_bytes())
    # Next stages replace only named starter files after archiving; other user files stay intact.
    new = series['stages'][state['stage'] + 1]
    for name in new['starterFiles']:
        if name not in stage['starterFiles'] and (root / name).exists(): raise ValueError(f'Keep existing {name}; move it before advancing.')
    for name, code in new['starterFiles'].items(): atomic(root / name, code.encode())
    state['stage'] += 1
    save_progress(root, state, series)
    print('Next: ' + new['title'] + '\n' + new['goal'] + '\nStart with morrowdock learn, then morrowdock guide.')


def learn(root, guided=False):
    state, series, stage = load(root)
    print('Learn · ' + stage['title'] + '\n' + stage['lesson'])
    print('Worked example: ' + stage['workedExample'])
    for n, step in enumerate(stage['walkthrough'],1): print(f'{n}. {step}')
    if guided:
        print('Guided practice · ' + stage['guidedPrompt'])
        for hint in stage['hints']: print('Hint: ' + hint)
        state.setdefault('practicedWithHelp', [])
        if stage['id'] not in state['practicedWithHelp']: state['practicedWithHelp'].append(stage['id'])
        save_progress(root,state,series)
    print('Learning and hints are free. Your own stage check comes when you are ready.')


def reveal(root):
    state, series, stage = load(root)
    if stage['id'] not in state.get('attempted', []):
        raise ValueError('Try a small version first. Use morrowdock learn or morrowdock guide for free teaching and hints.')
    state.setdefault('practicedWithHelp', [])
    if stage['id'] not in state['practicedWithHelp']: state['practicedWithHelp'].append(stage['id'])
    # Retain the neutral learning tag before exposing any solution. Never replace learner code.
    save_progress(root,state,series)
    target=root/'.morrowdock/reveals'/stage['id']; target.mkdir(parents=True,exist_ok=True)
    files_to_show=dict(stage['referenceFiles'])
    if stage['id']=='gpt-6': files_to_show.update(series.get('optionalRevealFiles', {}))
    for name, code in files_to_show.items():
        dest=target/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(code,encoding='utf-8')
    print('Optional worked solution: ' + str(target))
    print('Practiced with help. Study the explanation, close the reference, then rebuild your own version in src and check it. No progress is deducted.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build one small stage at a time. Local programs use your normal OS permissions.')
    parser.add_argument('command', choices=['init', 'learn', 'guide', 'reveal', 'check', 'next', 'qr'])
    parser.add_argument('series', nargs='?')
    parser.add_argument('--directory', type=Path, default=Path.cwd())
    parser.add_argument('--portfolio', action='store_true', help='Check local portfolio evidence as well as the current stage')
    args = parser.parse_args(argv)
    if args.portfolio and args.command != 'check': parser.error('--portfolio belongs to check')
    try:
        if args.command == 'init': initialize(args.series, args.directory)
        elif args.command == 'check': return check(args.directory, args.portfolio)
        elif args.command == 'next': advance(args.directory)
        elif args.command in ['learn','guide']: learn(args.directory,args.command=='guide')
        elif args.command == 'reveal': reveal(args.directory)
        else:
            import qrcode
            data = (args.directory / 'morrowdock-progress.json').read_text()
            qrcode.make(data).save(args.directory / 'morrowdock-progress.png')
            print('QR saved to morrowdock-progress.png. Files or AirDrop also work.')
        return 0
    except (ValueError, OSError, KeyError, StopIteration, ImportError) as e:
        print('Morrowdock: ' + str(e), file=sys.stderr); return 2

if __name__ == '__main__': sys.exit(main())
