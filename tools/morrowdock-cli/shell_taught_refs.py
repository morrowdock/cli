"""Reference implementations for the taught shell series (shell2). Host authoring data only.
Cumulative per stage, like gpt_taught_refs."""

TOKENIZE = '''
OPERATORS = ('>>', '|', '<', '>')

def tokenize(line: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    word, in_word, i = [], False, 0
    def finish():
        nonlocal word, in_word
        if in_word:
            tokens.append(('word', ''.join(word)))
        word, in_word = [], False
    while i < len(line):
        c = line[i]
        if c in ' \\t\\n':
            finish(); i += 1
        elif c == "'":
            end = line.find("'", i + 1)
            if end < 0:
                raise ValueError('Unterminated single quote')
            word.append(line[i + 1:end]); in_word = True; i = end + 1
        elif c == '"':
            i += 1; in_word = True
            while True:
                if i >= len(line):
                    raise ValueError('Unterminated double quote')
                c = line[i]
                if c == '"':
                    i += 1; break
                if c == '\\\\' and i + 1 < len(line) and line[i + 1] in '"\\\\':
                    word.append(line[i + 1]); i += 2
                else:
                    word.append(c); i += 1
        elif c == '\\\\':
            if i + 1 >= len(line):
                raise ValueError('Nothing to escape at end of line')
            word.append(line[i + 1]); in_word = True; i += 2
        else:
            op = next((o for o in OPERATORS if line.startswith(o, i)), None)
            if op:
                finish(); tokens.append(('op', op)); i += len(op)
            else:
                word.append(c); in_word = True; i += 1
    finish()
    return tokens

def parse(line: str) -> list[dict]:
    commands, current = [], {'argv': [], 'stdin': None, 'stdout': None, 'append': False}
    tokens = tokenize(line)
    if not tokens:
        return []
    i = 0
    while i < len(tokens):
        kind, text = tokens[i]
        if kind == 'word':
            current['argv'].append(text)
        elif text == '|':
            if not current['argv']:
                raise ValueError('Empty command before |')
            commands.append(current)
            current = {'argv': [], 'stdin': None, 'stdout': None, 'append': False}
        else:
            if i + 1 >= len(tokens) or tokens[i + 1][0] != 'word':
                raise ValueError(f'{text} needs a file name')
            target = tokens[i + 1][1]
            if text == '<':
                current['stdin'] = target
            else:
                current['stdout'] = target; current['append'] = text == '>>'
            i += 1
        i += 1
    if not current['argv']:
        raise ValueError('Empty command at end of line')
    commands.append(current)
    return commands
'''

BUILTINS = '''
import os
import subprocess

class ShellState:
    def __init__(self, cwd: str, env: dict | None = None):
        self.cwd = os.path.abspath(cwd)
        self.env = dict(env if env is not None else os.environ)
        self.status = 0
        self.exited = False

def run_command(argv: list[str], state: ShellState, stdin_text: str = '') -> tuple[int, str, str]:
    if not argv:
        raise ValueError('Nothing to run')
    name, args = argv[0], argv[1:]
    if name == 'cd':
        target = args[0] if args else state.env.get('HOME', state.cwd)
        path = os.path.abspath(os.path.join(state.cwd, target))
        if len(args) > 1 or not os.path.isdir(path):
            state.status = 1
            return 1, '', f'cd: {target}: no such directory\\n'
        state.cwd = path; state.status = 0
        return 0, '', ''
    if name == 'export':
        for pair in args:
            key, eq, value = pair.partition('=')
            if not eq or not key or not (key[0].isalpha() or key[0] == '_') or not all(ch.isalnum() or ch == '_' for ch in key):
                state.status = 1
                return 1, '', f'export: {pair}: use NAME=value\\n'
            state.env[key] = value
        state.status = 0
        return 0, '', ''
    if name == 'exit':
        code = int(args[0]) if args else state.status
        state.exited = True; state.status = code
        return code, '', ''
    try:
        p = subprocess.run(argv, cwd=state.cwd, env=state.env, input=stdin_text, capture_output=True, text=True, timeout=5, shell=False)
        state.status = p.returncode
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        state.status = 127
        return 127, '', f'{name}: command not found\\n'
    except PermissionError:
        state.status = 126
        return 126, '', f'{name}: permission denied\\n'
'''

PIPES = '''
def run_pipeline(commands: list[dict], state: ShellState) -> tuple[list[int], str]:
    """Start every stage at once, connected by OS pipes. Return each exit status and the last stage's
    stdout when it isn't redirected. Built-ins run only as a single command."""
    if not commands:
        return [], ''
    if len(commands) == 1 and commands[0]['argv'][0] in ('cd', 'export', 'exit'):
        cmd = commands[0]  # redirects open first, relative to the cwd the built-in starts in
        if cmd['stdin'] is not None:
            source = os.path.join(state.cwd, cmd['stdin']); open(source, 'rb').close()
        target = None if cmd['stdout'] is None else os.path.join(state.cwd, cmd['stdout'])
        if target is not None:
            open(target, 'a' if cmd['append'] else 'w', encoding='utf-8').close()
        code, out, err = run_command(cmd['argv'], state)
        if target is not None:
            with open(target, 'a', encoding='utf-8') as f:
                f.write(out)
            out = ''
        return [code], out
    procs, opened, previous, refused = [], [], None, {}
    try:
        for index, cmd in enumerate(commands):
            stdin = previous
            if cmd['stdin'] is not None:
                if previous is not None:
                    previous.close()
                stdin = open(os.path.join(state.cwd, cmd['stdin']), 'rb'); opened.append(stdin)
            last = index == len(commands) - 1
            if cmd['stdout'] is not None:
                stdout = open(os.path.join(state.cwd, cmd['stdout']), 'ab' if cmd['append'] else 'wb'); opened.append(stdout)
            else:
                stdout = subprocess.PIPE
            try:
                p = subprocess.Popen(cmd['argv'], cwd=state.cwd, env=state.env, stdin=stdin if stdin is not None else subprocess.DEVNULL,
                                     stdout=stdout, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                p = None; refused[index] = 127
            except PermissionError:
                p = None; refused[index] = 126
            if previous is not None:
                previous.close()  # the child has its own copy; keeping ours would hide EOF and SIGPIPE
            procs.append(p)
            previous = p.stdout if (p is not None and not last and cmd['stdout'] is None) else None
        tail = procs[-1]
        output = tail.stdout.read().decode() if (tail is not None and tail.stdout is not None) else ''
        codes = [refused[i] if p is None else p.wait(timeout=10) for i, p in enumerate(procs)]
    finally:
        for f in opened:
            f.close()
        for p in procs:
            if p is not None and p.stdout is not None:
                p.stdout.close()
    state.status = codes[-1]
    return codes, output
'''

MINIGIT = '''
import hashlib
import zlib
from pathlib import Path

def hash_object(kind: str, data: bytes) -> tuple[str, bytes]:
    if kind not in ('blob', 'tree', 'commit'):
        raise ValueError('Unknown object kind')
    raw = kind.encode() + b' ' + str(len(data)).encode() + b'\\0' + data
    return hashlib.sha1(raw).hexdigest(), raw

class MiniGit:
    def __init__(self, root: str):
        self.root = Path(root)
        (self.root / 'objects').mkdir(parents=True, exist_ok=True)
        (self.root / 'refs').mkdir(exist_ok=True)

    def put(self, kind: str, data: bytes) -> str:
        oid, raw = hash_object(kind, data)
        path = self.root / 'objects' / oid[:2] / oid[2:]
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(zlib.compress(raw))
        return oid

    def get(self, oid: str) -> tuple[str, bytes]:
        if len(oid) != 40 or any(c not in '0123456789abcdef' for c in oid):
            raise ValueError('Object ids are 40 lowercase hex characters')
        path = self.root / 'objects' / oid[:2] / oid[2:]
        if not path.exists():
            raise KeyError(oid)
        raw = zlib.decompress(path.read_bytes())
        header, _, data = raw.partition(b'\\0')
        kind, size = header.decode().split(' ')
        if int(size) != len(data) or hash_object(kind, data)[0] != oid:
            raise ValueError('Corrupt object')
        return kind, data

    def write_tree(self, files: dict[str, bytes]) -> str:
        entries = b''
        for name in sorted(files):
            if not name or '/' in name or name in ('.', '..') or '\\0' in name:
                raise ValueError('Flat file names only')
            blob = self.put('blob', files[name])
            entries += b'100644 ' + name.encode() + b'\\0' + bytes.fromhex(blob)
        return self.put('tree', entries)

    def commit(self, tree: str, parents: list[str], message: str, author: str) -> str:
        if self.get(tree)[0] != 'tree' or any(self.get(p)[0] != 'commit' for p in parents):
            raise ValueError('Commit a tree onto commit parents')
        lines = [f'tree {tree}'] + [f'parent {p}' for p in parents] + [f'author {author}', f'committer {author}', '', message]
        return self.put('commit', ('\\n'.join(lines) + '\\n').encode())

    def update_ref(self, name: str, new: str, expected_old: str | None) -> None:
        if not name or not all(c.isalnum() or c in '-_' for c in name):
            raise ValueError('Ref names use letters, digits, - and _')
        if self.get(new)[0] != 'commit':
            raise ValueError('Refs point at commits')
        path = self.root / 'refs' / name
        current = path.read_text().strip() if path.exists() else None
        if current != expected_old:
            raise ValueError('Ref moved since you read it')
        path.write_text(new + '\\n')

    def read_ref(self, name: str) -> str | None:
        path = self.root / 'refs' / name
        return path.read_text().strip() if path.exists() else None

    def log(self, name: str) -> list[str]:
        out, oid = [], self.read_ref(name)
        while oid:
            out.append(oid)
            headers = self.get(oid)[1].decode().split('\\n\\n', 1)[0]
            parents = [line[7:] for line in headers.split('\\n') if line.startswith('parent ')]
            oid = parents[0] if parents else None
        return out
'''

CUMULATIVE = [TOKENIZE, TOKENIZE + BUILTINS, TOKENIZE + BUILTINS + PIPES, TOKENIZE + BUILTINS + PIPES + MINIGIT]
