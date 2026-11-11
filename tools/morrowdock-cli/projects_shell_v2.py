"""Taught 'Build your own shell' (series shell2): original lessons, checks and wrong implementations.

Additive: the legacy `shell` series and its four stage identities stay byte-identical. Host
authoring data only. Stage checks 2–3 start real child processes (sys.executable) on the
learner's machine through the explicit local CLI; nothing here admits an app runtime.
"""
from textwrap import dedent

from shell_taught_refs import CUMULATIVE

HEADER = 'from __future__ import annotations\n'
PRELUDE = 'import os, sys, time, tempfile, random, shlex, subprocess\n'
PY = 'sys.executable'


def swap(source, old, new):
    """A named wrong implementation: the reference with one exact change. Fails loudly if `old` is absent."""
    if source.count(old) != 1:
        raise ValueError(f'mutant target not found exactly once: {old!r}')
    return source.replace(old, new)


def check(name, source):
    return dict(name=name, source=PRELUDE + dedent(source).strip() + '\n')


RUBRIC = [
    'Predict the tokens, statuses or object ids by hand before running.',
    'Show one input your own test covers that a simpler implementation gets wrong.',
    'Explain one bug, the counterexample that exposed it and the fix.',
    'Describe help used neutrally; authored examples are not your measured results.',
]

STAGES = [
    dict(
        title='1 · Tokenize and parse a command line yourself',
        goal='Write a quote-aware tokenizer and a parser for pipelines and redirects, with no shlex.',
        prerequisites='Python strings, indexing, while-loops, exceptions. Terminal Lab missions (quotes, pipes, > and >>) recommended.',
        lesson='''Before a shell runs anything it turns one line of text into words and operators. Get this wrong and the right program runs with the wrong arguments, which is how both bugs and security holes start. Write it yourself as a small state machine that reads one character at a time:
- Outside quotes, space/tab/newline ends a word. `|`, `<`, `>` and `>>` are operators and also end a word. Check `>>` before `>`.
- 'single quotes' keep everything literally, backslashes included, up to the next '.
- "double quotes" keep spaces. Inside them a backslash escapes only `"` and `\\`; any other backslash stays as a character.
- Outside quotes a backslash makes the next character literal. A backslash at the very end is an error.
- Quoted and unquoted pieces that touch join into one word: a"b c"\\ d is the single word `ab c d`. A quoted empty string "" is a real, empty argument, which is why you track "in a word" separately from "word text is non-empty".
- An unterminated quote is an error, not a guess.

Then parse the tokens into commands: split on `|`; `< file` sets stdin; `> file` or `>> file` sets stdout and whether to append. Reject an empty command (a leading, doubled or trailing pipe) and an operator with no file after it. A quoted '|' is a word, not an operator. Return plain dicts; executing comes in stage 3.

Misconception: “split on spaces, then strip the quotes.” It breaks the moment a quoted argument contains a space or a pipe.
Counterexample: echo "a | b" must be one command with the argument `a | b`.

You may compare with Python's shlex.split, which follows the same quoting rules this lesson uses, but your tokenizer must not call it. The hidden checks use shlex as an independent oracle on thousands of random lines.

Guided activity A: tokenize a"b c"\\ d and echo a\\b "q\\"x\\y" by hand. Guided activity B: parse sort < data.txt | uniq -c >> counts.txt into two command dicts.
On your own: write a test with a quoted operator and an empty argument before running the hidden checks.''',
        starter='''# Stage 1. No shlex: write the state machine yourself.

def tokenize(line: str) -> list[tuple[str, str]]:
    # Return ('word', text) and ('op', '|' / '<' / '>' / '>>') tokens in order.
    raise NotImplementedError

def parse(line: str) -> list[dict]:
    # Return [{'argv': [...], 'stdin': None or name, 'stdout': None or name, 'append': bool}, ...]
    raise NotImplementedError''',
        worked='''echo 'two words' | tr a-z A-Z > out.txt → [word echo, word "two words", op |, word tr, word a-z, word A-Z, op >, word out.txt]. a"b c"\\ d → one word "ab c d". echo a\\b "q\\"x\\y" → echo, ab, q"x\\y: outside quotes \\b becomes b; inside double quotes \\" becomes " and \\y stays as a backslash plus y. grep -c x<in>>log → grep, -c, x, <, in, >>, log: operators split words even without spaces. "" → one empty word. Parsing sort < data.txt | uniq -c >> counts.txt gives [{argv: [sort], stdin: data.txt}, {argv: [uniq, -c], stdout: counts.txt, append: True}].''',
        walkthrough=['Write a loop with an index i; handle whitespace, both quote kinds, backslash, operators and ordinary characters.',
                     'Keep an in_word flag so "" produces an empty argument.',
                     'Parse: walk the tokens, start a new command at |, consume the file name after a redirect.',
                     'Raise ValueError for unterminated quotes, a trailing backslash, empty commands and missing file names.'],
        hints=['Match ">>" before ">" so the longer operator wins.',
               'Inside double quotes, only \\" and \\\\ are escapes.'],
        checks=[
            check('worked example tokens and parse', '''
assert tokenize("echo 'two words' | tr a-z A-Z > out.txt") == [('word','echo'),('word','two words'),('op','|'),('word','tr'),('word','a-z'),('word','A-Z'),('op','>'),('word','out.txt')]
assert tokenize('a"b c"\\\\ d') == [('word','ab c d')]
assert parse('sort < data.txt | uniq -c >> counts.txt') == [{'argv':['sort'],'stdin':'data.txt','stdout':None,'append':False},{'argv':['uniq','-c'],'stdin':None,'stdout':'counts.txt','append':True}]
'''),
            check('agrees with POSIX shlex on random quoted words', '''
rng = random.Random(20261103)
alphabet = ['a', 'b', ' ', "'", '"', '\\\\', 'c d', '\\t']
for _ in range(4000):
    line = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 10)))
    try: expected = shlex.split(line)
    except ValueError: expected = None
    try: got = [t for k, t in tokenize(line)] if all(k == 'word' for k, t in tokenize(line)) else 'op'
    except ValueError: got = None
    assert got == expected, (line, got, expected)
'''),
            check('operators, quoted operators and empty arguments', '''
assert tokenize('grep -c x<in>>log') == [('word','grep'),('word','-c'),('word','x'),('op','<'),('word','in'),('op','>>'),('word','log')]
assert parse('echo "a | b" \\'>\\' ""') == [{'argv':['echo','a | b','>',''],'stdin':None,'stdout':None,'append':False}]
assert parse('cat>x') == [{'argv':['cat'],'stdin':None,'stdout':'x','append':False}]
assert parse('echo hi >> a > b')[0]['stdout'] == 'b' and parse('echo hi >> a > b')[0]['append'] is False  # the last redirect wins
assert parse('   ') == [] and parse('a|b|c')[2]['argv'] == ['c']
'''),
            check('malformed lines are rejected', '''
for line in ['| cat', 'ls |', 'a || b', 'sort <', 'echo > | x', 'echo "open', "it's", 'tail\\\\']:
    try: parse(line)
    except ValueError: pass
    else: raise AssertionError('reject: ' + line)
'''),
        ],
        mutants=[
            ('double quotes escape everything', 'tokenize', swap(CUMULATIVE[0], r"""and line[i + 1] in '"\\'""", '')),
            ('quoted empty argument dropped', 'tokenize', swap(CUMULATIVE[0], "            i += 1; in_word = True\n", "            i += 1\n")),
            ('>> read as two >', 'tokenize', swap(CUMULATIVE[0], "OPERATORS = ('>>', '|', '<', '>')", "OPERATORS = ('|', '<', '>', '>>')")),
            ('single quotes process backslashes', 'tokenize', swap(CUMULATIVE[0], "word.append(line[i + 1:end]); in_word = True", r"word.append(line[i + 1:end].replace('\\', '')); in_word = True")),
            ('append flag sticks after >>', 'parse', swap(CUMULATIVE[0], "current['append'] = text == '>>'", "current['append'] = current['append'] or text == '>>'")),
        ]),
    dict(
        title='2 · Run a process, and know what must be a built-in',
        goal='Run external programs with an argument list, cwd and environment, and implement cd/export/exit as built-ins with shell-style statuses.',
        prerequisites='Stage 1; subprocess.run; os.path; what a process, cwd and environment variable are.',
        lesson='''Every external command runs as a separate child process. The child gets a copy of the shell's current directory and environment, and it can change its own copy, never yours. That single fact decides what a shell must do itself:
- cd must be a built-in. A child process that changes directory dies with that change; the shell's own cwd never moves. Your shell keeps its cwd in a ShellState and passes it to every child (cwd=state.cwd).
- export must be a built-in, for the same reason: update state.env and pass env=state.env to children. Names start with a letter or _, then letters, digits or _.
- exit marks the shell as finished with a status; with no argument it reuses the last status.

Pass the argument list straight to subprocess.run with shell=False. The words from stage 1 reach the program exactly as parsed, with no second round of interpretation. That is the defence against an argument like `; rm -rf ~` being run as code.

Statuses are the shell's return values: 0 means success, anything else is a failure code chosen by the program. Shells use 127 for “command not found” and 126 for “found but not executable”, so produce those instead of crashing. Record the last status in state.status, which is what $? reports in a real shell. cd to a missing directory prints a message and returns 1, leaving the cwd unchanged.

Misconception: “I'll run cd through subprocess like everything else.” The check below starts a child that changes directory and shows your shell's cwd is untouched.

Guided activity A: predict the output of a child that prints os.getcwd() after cd sub. Guided activity B: export GREETING=hi, then run a child that prints it; run another without env=, and explain the difference.
On your own: the hidden checks use statuses, a missing program and invalid names.''',
        starter='''# Keep stage 1 above. Stage 2:
import os
import subprocess

class ShellState:
    def __init__(self, cwd: str, env: dict | None = None):
        raise NotImplementedError

def run_command(argv: list[str], state: ShellState, stdin_text: str = '') -> tuple[int, str, str]:
    # Built-ins cd, export, exit change state; everything else runs as a child. Return (status, stdout, stderr).
    raise NotImplementedError''',
        worked='''Start with state.cwd = /tmp/x containing a folder sub. run_command(['cd','sub']) gives (0,'','') and state.cwd = /tmp/x/sub. A child running os.getcwd() prints /tmp/x/sub, because it inherited the cwd. A child running os.chdir('/') changes only itself, so state.cwd is still /tmp/x/sub. run_command(['cd','missing']) gives status 1 with a message, and the cwd is unchanged. ['export','GREETING=hi'] then a child printing os.environ['GREETING'] prints hi. ['no-such-program'] gives (127, '', 'no-such-program: command not found\\n'). ['exit','3'] sets state.exited and state.status = 3.''',
        walkthrough=['Store cwd (absolute), a copy of the environment, the last status and an exited flag in ShellState.',
                     'Dispatch on argv[0]: cd, export, exit are handled in your process.',
                     'Run others with subprocess.run(argv, cwd=..., env=..., input=..., capture_output=True, text=True, timeout=..., shell=False).',
                     'Map FileNotFoundError to 127 and PermissionError to 126; always update state.status.'],
        hints=['os.path.abspath(os.path.join(state.cwd, target)) handles both relative and absolute targets.',
               'Copy the environment dict so export doesn\'t change your Python process.'],
        checks=[
            check('cd is a built-in and children inherit cwd', f'''
with tempfile.TemporaryDirectory() as d:
    os.mkdir(os.path.join(d, 'sub'))
    s = ShellState(d)
    assert run_command(['cd', 'sub'], s) == (0, '', '') and os.path.samefile(s.cwd, os.path.join(d, 'sub'))
    code, out, err = run_command([{PY}, '-c', 'import os; print(os.getcwd())'], s)
    assert code == 0 and os.path.samefile(out.strip(), s.cwd)
'''),
            check('a child cannot move the shell, and cd errors keep the cwd', f'''
with tempfile.TemporaryDirectory() as d:
    s = ShellState(d); before = s.cwd
    assert run_command([{PY}, '-c', 'import os; os.chdir(os.sep)'], s)[0] == 0 and s.cwd == before
    code, out, err = run_command(['cd', 'missing'], s)
    assert code == 1 and err and s.cwd == before and s.status == 1
    os.makedirs(os.path.join(d, 'dock', 'crane'))
    assert run_command(['cd', os.path.join('dock', 'crane')], s)[0] == 0 and run_command(['cd', '..'], s)[0] == 0
    code, out, _ = run_command([{PY}, '-c', 'import os; print(os.getcwd())'], s)
    assert os.path.samefile(s.cwd, os.path.join(d, 'dock')) and os.path.samefile(out.strip(), os.path.join(d, 'dock'))
    home = ShellState(os.path.join(d, 'dock', 'crane'), {{'HOME': d}})
    assert run_command(['cd'], home)[0] == 0 and os.path.samefile(home.cwd, d)  # cd with no argument goes HOME
'''),
            check('export reaches children; bad names refused', f'''
with tempfile.TemporaryDirectory() as d:
    s = ShellState(d, {{'PATH': os.environ.get('PATH', ''), 'SYSTEMROOT': os.environ.get('SYSTEMROOT', '')}})
    assert run_command(['export', 'GREETING=hi there'], s)[0] == 0
    code, out, _ = run_command([{PY}, '-c', 'import os; print(os.environ["GREETING"])'], s)
    assert out.strip() == 'hi there' and 'GREETING' not in os.environ
    for bad in ['1X=a', 'NOEQUALS', '=v', 'A-B=c']:
        assert run_command(['export', bad], s)[0] == 1
    assert s.status == 1 and run_command(['export', 'OK=1'], s)[0] == 0 and s.status == 0
'''),
            check('statuses: exit codes, not found, exit built-in, literal arguments', f'''
with tempfile.TemporaryDirectory() as d:
    s = ShellState(d)
    code, out, _ = run_command([{PY}, '-c', 'import sys; print(sys.argv[1]); sys.exit(4)', '; echo hacked | x'], s)
    assert code == 4 and out.strip() == '; echo hacked | x' and s.status == 4
    os.mkdir(os.path.join(d, 'bin'))
    code, out, err = run_command([os.path.join(d, 'bin')], s)  # found, but a directory cannot be executed
    assert code == 126 and err and s.status == 126
    with open(os.path.join(d, 'tool'), 'w') as f: f.write('#!/bin/sh\\necho hi\\n')
    if os.name != 'nt':
        os.chmod(os.path.join(d, 'tool'), 0o644)
        s.status = 0
        code, out, err = run_command([os.path.join(d, 'tool')], s)  # found, but not marked executable
        assert code == 126 and err and s.status == 126
    assert run_command(['cd', 'tool'], s)[0] == 1 and os.path.samefile(s.cwd, d)  # cd onto a file is refused
    code, out, err = run_command(['definitely-not-a-program-7731'], s)
    assert code == 127 and 'not found' in err and s.status == 127
    assert run_command(['exit'], s)[0] == 127 and s.exited
    s2 = ShellState(d); run_command(['exit', '3'], s2); assert s2.exited and s2.status == 3
'''),
        ],
        mutants=[
            ('cd in a child process', 'run_command', CUMULATIVE[1].split('class ShellState')[0] + 'class ShellState' + CUMULATIVE[1].split('class ShellState')[1].replace("    if name == 'cd':", "    if name == 'cd-builtin-disabled':")),
            ('environment not passed', 'run_command', swap(CUMULATIVE[1], "cwd=state.cwd, env=state.env, input", "cwd=state.cwd, input")),
            ('not found crashes', 'run_command', swap(CUMULATIVE[1], "    except FileNotFoundError:\n        state.status = 127\n        return 127, '', f'{name}: command not found\\n'\n", "")),
            ('children ignore the shell cwd', 'run_command', swap(CUMULATIVE[1], "cwd=state.cwd, env=state.env, input", "env=state.env, input")),
            ('not executable crashes', 'run_command', swap(CUMULATIVE[1], "    except PermissionError:\n", "    except NotImplementedError:\n")),
            ('not executable reported as 127', 'run_command', swap(CUMULATIVE[1], "        state.status = 126\n        return 126, ", "        state.status = 127\n        return 127, ")),
            ('cd with no argument stays put', 'run_command', swap(CUMULATIVE[1], "target = args[0] if args else state.env.get('HOME', state.cwd)", "target = args[0] if args else '.'")),
            ('export keeps the old status', 'run_command', swap(CUMULATIVE[1], "            state.env[key] = value\n        state.status = 0\n", "            state.env[key] = value\n")),
        ]),
    dict(
        title='3 · Real pipes and redirects',
        goal='Run every pipeline stage at once, connected by OS pipes, with < > >> redirects relative to the shell cwd.',
        prerequisites='Stages 1–2; file objects; subprocess.Popen; what end-of-file means for a reader.',
        lesson='''The Terminal Lab passes text from one command to the next after each finishes. A real shell starts every stage at once and connects them with OS pipes: stage 1's stdout is stage 2's stdin, and data flows while both run. Three consequences:
1. Speed and memory: three stages that each take a second finish in about one second, not three, and a gigabyte can stream through without being held in memory.
2. End-of-file: a reader sees EOF only when every copy of the pipe's write end is closed. Popen gives the child its own copy, so after starting the next stage your shell must close its own copy of the previous pipe. Forget, and a reader that waits for EOF hangs.
3. Early exit: if a later stage quits early (like head), the producer's next write fails with a broken pipe and it stops. That only happens if no one else, your shell included, still holds the read end. Keeping your copy open makes the producer block once the pipe buffer (often 64 KiB) fills. This is the most common pipeline hang, and a hidden check times it.

Redirects replace a stage's stdin or stdout with a file, opened relative to state.cwd (not your Python process's cwd): `<` reads, `>` truncates, `>>` appends. A redirected stdout isn't piped onward, so the next stage gets no input from it. Wait for every process, return all statuses, and set state.status to the last stage's status, which is what POSIX shells report by default. A missing program in the middle gets status 127, and the stages around it still finish. Its upstream neighbour may then hit a broken pipe, because nobody reads its output, so that stage's status depends on timing, exactly as in a real shell. Don't write tests that assume it.

Misconception: “subprocess.run each stage and feed the output forward is the same thing.” It gives the same text for small inputs, but it runs stages one after another, holds everything in memory and can't stop early.

Guided activity A: draw the file descriptors for a | b | c and mark which ends your shell must close, and when. Guided activity B: time three one-second stages, first sequentially, then concurrently.
On your own: the hidden checks include early exit, redirects after cd and a missing middle stage.''',
        starter='''# Keep stages 1–2 above. Stage 3:

def run_pipeline(commands: list[dict], state: ShellState) -> tuple[list[int], str]:
    # commands come from parse(). Return (every status, final stdout if not redirected).
    raise NotImplementedError''',
        worked='''parse('python -c "print(6)" | python -c "import sys; print(int(sys.stdin.read())*7)"') runs both at once. The second reads until EOF, which arrives when the first exits and your shell has closed its copy of the pipe. The result is ([0, 0], '42\\n'). With state.cwd = /tmp/x, the line `python -c "print(1)" > a.txt` creates /tmp/x/a.txt holding "1\\n". A second `>>` makes it "1\\n1\\n". A producer writing 10 MB into a consumer that reads one line and exits finishes in well under a second: the producer gets a broken pipe.''',
        walkthrough=['For each stage choose stdin (previous pipe, < file, or nothing) and stdout (PIPE, > or >> file).',
                     'Start it with Popen, then close your copy of the previous pipe.',
                     'Read the last stage\'s stdout to the end, then wait for every process.',
                     'Close any files you opened; return the statuses and the output; set state.status.'],
        hints=['After Popen(..., stdin=previous.stdout), call previous.stdout.close() in the parent.',
               'Open redirect files with os.path.join(state.cwd, name).'],
        checks=[
            check('two stages connected by a pipe', f'''
with tempfile.TemporaryDirectory() as d:
    s = ShellState(d)
    cmds = [{{'argv':[{PY},'-c','print(6)'],'stdin':None,'stdout':None,'append':False}},
            {{'argv':[{PY},'-c','import sys; print(int(sys.stdin.read())*7)'],'stdin':None,'stdout':None,'append':False}}]
    codes, out = run_pipeline(cmds, s)
    assert codes == [0, 0] and out.strip() == '42' and s.status == 0
'''),
            check('stages run at the same time', f'''
with tempfile.TemporaryDirectory() as d:
    s = ShellState(d)
    stage = {{'argv':[{PY},'-c','import sys,time; time.sleep(1); sys.stdout.write(sys.stdin.read())'],'stdin':None,'stdout':None,'append':False}}
    start = time.time(); codes, out = run_pipeline([dict(stage), dict(stage), dict(stage)], s)
    assert codes == [0,0,0] and time.time() - start < 2.5, 'three one-second stages should overlap'
'''),
            check('an early-exiting reader stops the producer (parent closes its pipe copy)', f'''
producer = "import os,sys,threading; threading.Timer(6, lambda: os._exit(9)).start()\\ntry:\\n    [sys.stdout.write('x'*999+'\\\\n') for _ in range(20000)]; sys.stdout.flush()\\nexcept BrokenPipeError:\\n    os._exit(1)\\nos._exit(0)"
with tempfile.TemporaryDirectory() as d:
    s = ShellState(d)
    cmds = [{{'argv':[{PY},'-c',producer],'stdin':None,'stdout':None,'append':False}},
            {{'argv':[{PY},'-c','import sys; print(len(sys.stdin.readline()))'],'stdin':None,'stdout':None,'append':False}}]
    start = time.time(); codes, out = run_pipeline(cmds, s)
    assert out.strip() == '1000' and codes[1] == 0 and codes[0] != 9 and time.time() - start < 4
'''),
            check('redirects follow the shell cwd; statuses; missing middle stage', f'''
back = os.getcwd()
with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as elsewhere:
    os.chdir(elsewhere)  # the Python process stands somewhere else on purpose
    try:
        os.mkdir(os.path.join(d, 'w')); s = ShellState(d); run_command(['cd', 'w'], s)
        say = lambda t: {{'argv':[{PY},'-c',f'print({{t!r}})'],'stdin':None,'stdout':'log.txt','append':False}}
        run_pipeline([say('one')], s); first = dict(say('two')); first['append'] = True; run_pipeline([first], s)
        assert open(os.path.join(d, 'w', 'log.txt')).read() == 'one\\ntwo\\n'
        codes, out = run_pipeline([{{'argv':[{PY},'-c','import sys; print(sys.stdin.read().count(chr(10)))'],'stdin':'log.txt','stdout':None,'append':False}}], s)
        assert out.strip() == '2' and not os.path.exists(os.path.join(elsewhere, 'log.txt'))
        run_pipeline([say('three')], s)  # > truncates
        assert open(os.path.join(d, 'w', 'log.txt')).read() == 'three\\n'
        codes, out = run_pipeline([{{'argv':[{PY},'-c','import sys; sys.exit(5)'],'stdin':None,'stdout':None,'append':False}}, {{'argv':[{PY},'-c','import sys; sys.stdin.read()'],'stdin':None,'stdout':None,'append':False}}], s)
        assert codes == [5, 0] and s.status == 0  # every stage is waited for and reported
        codes, out = run_pipeline([{{'argv':[{PY},'-c','print(1)'],'stdin':None,'stdout':None,'append':False}}, {{'argv':['no-such-program-7731'],'stdin':None,'stdout':None,'append':False}}, {{'argv':[{PY},'-c','import sys; sys.stdin.read(); sys.exit(3)'],'stdin':None,'stdout':None,'append':False}}], s)
        assert len(codes) == 3 and codes[1:] == [127, 3] and s.status == 3  # codes[0] races with a broken pipe: nothing reads it
        os.mkdir(os.path.join(d, 'w', 'bin'))  # found, but a directory cannot be executed
        codes, out = run_pipeline([{{'argv':[os.path.join(d, 'w', 'bin')],'stdin':None,'stdout':None,'append':False}}, {{'argv':[{PY},'-c','import sys; sys.stdin.read(); sys.exit(4)'],'stdin':None,'stdout':None,'append':False}}], s)
        assert codes == [126, 4] and s.status == 4
        if os.name != 'nt':
            with open(os.path.join(d, 'w', 'tool'), 'w') as f: f.write('#!/bin/sh\\necho hi\\n')
            os.chmod(os.path.join(d, 'w', 'tool'), 0o644)
            codes, out = run_pipeline([{{'argv':[os.path.join(d, 'w', 'tool')],'stdin':None,'stdout':None,'append':False}}], s)
            assert codes == [126] and s.status == 126  # found, but not marked executable
        codes, out = run_pipeline([{{'argv':['export','A=b'],'stdin':None,'stdout':'log.txt','append':False}}], s)
        assert codes == [0] and out == '' and s.env['A'] == 'b' and open(os.path.join(d, 'w', 'log.txt')).read() == ''  # a built-in's > still truncates
        run_pipeline([{{'argv':['cd','..'],'stdin':None,'stdout':'made.txt','append':True}}], s)
        assert os.path.exists(os.path.join(d, 'w', 'made.txt')) and os.path.samefile(s.cwd, d)  # opened before cd ran
    finally:
        os.chdir(back)  # a folder that is still the cwd can't be deleted on Windows
'''),
        ],
        mutants=[
            ('sequential buffered pipeline', 'run_pipeline', CUMULATIVE[1] + '''
def run_pipeline(commands, state):
    text, codes = '', []
    for cmd in commands:
        stdin_text = open(os.path.join(state.cwd, cmd['stdin'])).read() if cmd['stdin'] else text
        code, out, err = run_command(cmd['argv'], state, stdin_text)
        if cmd['stdout']:
            open(os.path.join(state.cwd, cmd['stdout']), 'a' if cmd['append'] else 'w').write(out); out = ''
        text = out; codes.append(code)
    return codes, text
'''),
            ('parent keeps its pipe copy', 'run_pipeline', swap(CUMULATIVE[2], "            if previous is not None:\n                previous.close()  # the child", "            if False:\n                previous.close()  # the child")),
            ('redirects relative to the Python process', 'run_pipeline', swap(CUMULATIVE[2], "open(os.path.join(state.cwd, cmd['stdout'])", "open(cmd['stdout']").replace("open(os.path.join(state.cwd, cmd['stdin'])", "open(cmd['stdin']")),
            ('first status reported', 'run_pipeline', swap(CUMULATIVE[2], "    state.status = codes[-1]\n", "    state.status = codes[0]\n")),
            ('append always', 'run_pipeline', swap(CUMULATIVE[2], "'ab' if cmd['append'] else 'wb'", "'ab'")),
            ('wait only the last process', 'run_pipeline', swap(CUMULATIVE[2], "codes = [refused[i] if p is None else p.wait(timeout=10) for i, p in enumerate(procs)]", "codes = [refused[i] if p is None else (p.wait(timeout=10) if p is procs[-1] else 0) for i, p in enumerate(procs)]")),
            ('non-executable stage crashes', 'run_pipeline', swap(CUMULATIVE[2], "            except PermissionError:\n                p = None; refused[index] = 126\n", "")),
            ('built-in ignores its redirects', 'run_pipeline', swap(CUMULATIVE[2], "        target = None if cmd['stdout'] is None else os.path.join(state.cwd, cmd['stdout'])\n", "        target = None\n")),
        ]),
    dict(
        title='4 · A mini Git object store that real Git can read',
        goal='Store blobs, trees and commits in Git\'s own object format, move refs safely, and walk history.',
        prerequisites='Stages 1–3 (for the demo shell); bytes vs str; hashlib; zlib. Git Lab lessons 1–4 recommended.',
        lesson='''The Git Lab showed history as snapshots and branches as movable labels. Underneath, Git is a content-addressed store: every object is saved under the hash of its own bytes, so the same content always has the same id, and any change makes a new id.

Git's object format, which you'll match byte for byte:
- Header: kind, a space, the payload length in decimal, then a NUL byte. id = SHA-1(header + payload) as 40 hex characters. Store zlib-compressed bytes at objects/<first 2 hex>/<remaining 38>.
- blob: the file's bytes. b'hello\\n' must give ce013625030ba8dba906f756967f9e9ca394464a, which is what `git hash-object` prints.
- tree (flat, files only here): for each name in sorted order, b'100644 ' + name + b'\\0' + the 20 raw bytes of the blob id (not hex).
- commit: lines `tree <id>`, one `parent <id>` per parent, `author <name> <email> <unix time> <tz>`, the same committer line, a blank line, the message, and a final newline.
Get any byte wrong (unsorted names, hex instead of raw bytes, a missing length) and your ids stop matching Git's. The visible check compares against ids produced by real Git.

Reading an object back, recompute its hash and refuse if it doesn't match: content addressing is what lets Git detect corruption. Objects never change; refs are small files naming a commit. Moving a ref safely is compare-and-swap: update it only if it still holds the value you read, otherwise refuse, because someone else moved it. Real Git does the same check when you push.

Misconception: “changing one file rewrites the whole history.” It changes that file's blob, the tree and the new commit. The other file's blob id is identical, and old commits are untouched.

Guided activity A: hash b'hello\\n' by hand-assembling the header, and compare. Guided activity B: commit {a.txt, b.txt}, change b.txt, commit again, and list which ids changed.
On your own: the hidden checks tamper with a stored object and race two ref updates. Stretch: if git is installed, copy your objects into a fresh repository's .git/objects and run `git fsck --strict` and `git cat-file -p <id>`.''',
        starter='''# Keep stages 1–3 above. Stage 4:
import hashlib
import zlib
from pathlib import Path

def hash_object(kind: str, data: bytes) -> tuple[str, bytes]:
    raise NotImplementedError

class MiniGit:
    def __init__(self, root: str):
        raise NotImplementedError
    def put(self, kind: str, data: bytes) -> str:
        raise NotImplementedError
    def get(self, oid: str) -> tuple[str, bytes]:
        raise NotImplementedError
    def write_tree(self, files: dict[str, bytes]) -> str:
        raise NotImplementedError
    def commit(self, tree: str, parents: list[str], message: str, author: str) -> str:
        raise NotImplementedError
    def update_ref(self, name: str, new: str, expected_old: str | None) -> None:
        raise NotImplementedError
    def read_ref(self, name: str) -> str | None:
        raise NotImplementedError
    def log(self, name: str) -> list[str]:
        raise NotImplementedError''',
        worked='''hash_object('blob', b'hello\\n'): the header is b'blob 6\\x00', so the id is sha1(b'blob 6\\x00hello\\n') = ce013625030ba8dba906f756967f9e9ca394464a. Files {a.txt: b'x\\n', b.txt: b'y\\n'} give tree 5845e1ff32aed0df732e1c10d8ac578f6c172c61, the same id `git write-tree` prints. Committing that tree with no parents, message “First snapshot” and author “Ada <ada@example.invalid> 1700000000 +0000” gives commit 798de14458ec55bad3b06fe5abfa1e1cc5c05e59, which `git commit-tree` also prints. `git fsck --strict` on a copy of the store accepts it.''',
        walkthrough=['hash_object: build the header, hash, return (id, raw bytes).',
                     'put/get: compress and store under objects/aa/…; on get, decompress, check the length and recompute the hash.',
                     'write_tree with sorted names and raw 20-byte ids; commit with exact line format.',
                     'update_ref with compare-and-swap; log walks first parents from a ref.'],
        hints=['bytes.fromhex(blob_id) gives the 20 raw bytes a tree entry needs.',
               'Sort names before building the tree so equal snapshots have equal ids.'],
        checks=[
            check('ids match real Git', '''
assert hash_object('blob', b'hello\\n')[0] == 'ce013625030ba8dba906f756967f9e9ca394464a'
with tempfile.TemporaryDirectory() as d:
    g = MiniGit(d)
    t = g.write_tree({'b.txt': b'y\\n', 'a.txt': b'x\\n'})
    assert t == '5845e1ff32aed0df732e1c10d8ac578f6c172c61'
    assert g.commit(t, [], 'First snapshot', 'Ada <ada@example.invalid> 1700000000 +0000') == '798de14458ec55bad3b06fe5abfa1e1cc5c05e59'
'''),
            check('one change, which ids move; history walk', '''
with tempfile.TemporaryDirectory() as d:
    g = MiniGit(d); who = 'Kofi <kofi@example.invalid> 1700000100 +0000'
    t1 = g.write_tree({'notes': b'tide\\n', 'crew': b'ada\\n'}); c1 = g.commit(t1, [], 'one', who)
    t2 = g.write_tree({'notes': b'tide\\n', 'crew': b'ada, kofi\\n'}); c2 = g.commit(t2, [c1], 'two', who)
    blob = lambda tree, name: g.get(tree)[1].split(name.encode() + b'\\0', 1)[1][:20].hex()
    assert blob(t1, 'notes') == blob(t2, 'notes') and blob(t1, 'crew') != blob(t2, 'crew') and t1 != t2
    assert g.put('blob', b'tide\\n') == blob(t1, 'notes') and g.write_tree({'crew': b'ada\\n', 'notes': b'tide\\n'}) == t1
    g.update_ref('main', c1, None); g.update_ref('main', c2, c1)
    assert g.log('main') == [c2, c1] and g.get(c2)[0] == 'commit' and b'parent ' + c1.encode() in g.get(c2)[1]
    lone = g.commit(t2, [], 'parent ' + c1, who)  # a message line that looks like a header
    g.update_ref('side', lone, None)
    assert g.log('side') == [lone], 'parents come from the headers, not the message'
'''),
            check('corruption is detected; bad input refused', '''
with tempfile.TemporaryDirectory() as d:
    g = MiniGit(d); oid = g.put('blob', b'precious\\n')
    path = os.path.join(d, 'objects', oid[:2], oid[2:])
    import zlib as _z; open(path, 'wb').write(_z.compress(b'blob 9\\x00tampered\\n'))
    try: g.get(oid)
    except ValueError: pass
    else: raise AssertionError('a changed object must not pass as the original')
    for bad in ['XYZ', 'ab' * 19, oid.upper()]:
        try: g.get(bad)
        except (ValueError, KeyError): pass
        else: raise AssertionError('bad id ' + bad)
    for name in ['a/b', '', '..']:
        try: g.write_tree({name: b'x'})
        except ValueError: pass
        else: raise AssertionError('bad name ' + repr(name))
'''),
            check('refs move only by compare-and-swap, and only to commits', '''
with tempfile.TemporaryDirectory() as d:
    g = MiniGit(d); who = 'Ada <ada@example.invalid> 1700000000 +0000'
    t = g.write_tree({'f': b'1\\n'}); a = g.commit(t, [], 'a', who); b = g.commit(t, [a], 'b', who); c = g.commit(t, [a], 'c', who)
    g.update_ref('main', a, None)
    try: g.update_ref('main', b, None)
    except ValueError: pass
    else: raise AssertionError('creating a ref that already exists must be refused')
    assert g.read_ref('main') == a
    g.update_ref('main', b, a)
    try: g.update_ref('main', c, a)
    except ValueError: pass
    else: raise AssertionError('stale expected value must be refused')
    assert g.read_ref('main') == b
    for name, target in [('bad name', b), ('main', t)]:
        try: g.update_ref(name, target, b)
        except ValueError: pass
        else: raise AssertionError('refuse ' + name)
    assert g.read_ref('missing') is None and g.log('missing') == []
'''),
        ],
        mutants=[
            ('header without length', 'hash_object', swap(CUMULATIVE[3], r"raw = kind.encode() + b' ' + str(len(data)).encode() + b'\0' + data", r"raw = kind.encode() + b'\0' + data")),
            ('tree names unsorted', 'write_tree', swap(CUMULATIVE[3], "for name in sorted(files):", "for name in files:")),
            ('get trusts the bytes', 'get', swap(CUMULATIVE[3], "if int(size) != len(data) or hash_object(kind, data)[0] != oid:", "if False:")),
            ('ref update without compare', 'update_ref', swap(CUMULATIVE[3], "if current != expected_old:", "if False:")),
            ('create overwrites an existing ref', 'update_ref', swap(CUMULATIVE[3], "if current != expected_old:", "if expected_old is not None and current != expected_old:")),
            ('log reads parents from the message', 'log', swap(CUMULATIVE[3], "            parents = [line[7:] for line in headers.split('\\n') if line.startswith('parent ')]\n", "            parents = [line[7:] for line in self.get(oid)[1].decode().split('\\n') if line.startswith('parent ')]\n")),
        ]),
]

README_TEMPLATE = '''# <your shell name>: a POSIX-style shell and a Git-compatible object store, from scratch

> One line: what it does and what you verified (e.g. “my object ids match `git hash-object`/`git write-tree`; `git fsck --strict` accepts my store”). Only claims you checked yourself.

![demo](docs/demo.cast) <!-- your own asciinema recording: quoting, a pipeline, a redirect, an error status, two mini-git commits -->

## Quickstart
```sh
python -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
pytest
python -m <your_module>     # starts your shell
```

## What I built
| Stage | Component | My tests prove | Commit(s) |
| --- | --- | --- | --- |
| 1 | Tokenizer + parser (no shlex) | agrees with POSIX shlex on random lines; quoted operators stay words | |
| 2 | Processes + built-ins | cd/export are built-ins; 126/127 statuses; literal arguments | |
| 3 | Pipes + redirects | stages overlap; early exit doesn't hang; redirects follow cwd | |
| 4 | Object store + refs | ids equal real Git's; corruption detected; compare-and-swap refs | |

## Architecture
```mermaid
flowchart LR
  L[line] --> T[tokenize] --> P[parse] --> R{built-in?}
  R -- yes --> S[ShellState: cwd, env, status]
  R -- no --> X["Popen stages joined by OS pipes, redirects"]
  G[minigit] --> O[(objects/aa/bb… zlib)] & F[refs/ compare-and-swap]
```

## Design decisions and trade-offs
- Why the tokenizer is a hand-written state machine; which POSIX features I left out (globbing, $VAR expansion, &&, job control).
- Why the shell closes its pipe copies; what I measured when I didn't.
- SHA-1 for compatibility with Git, and its known collision weakness (real Git is moving to SHA-256).

## Results (measured: fill in)
- Pipeline throughput on <your machine> for <size> bytes; time for three 1-second stages.
- Real Git interoperability commands I ran and their output.

## Limits
- Not a security sandbox; no job control; flat trees only; no packfiles.

## What I learned / Next steps
- One bug a test caught, and the fix. See ROADMAP.md (job control, subdirectory trees, `checkout`).

License: <your choice, e.g. MIT>.
'''

FOLLOW_ON = '''# Try without a supplied solution: checkout and diff for your mini Git

Add two commands to your shell, backed by your object store: `checkout <ref> <dir>` writes a commit's files into an empty directory (refuse a non-empty one), and `diff <ref-a> <ref-b>` prints added, removed and changed file names between two commits' trees. Before implementing, write tests: a round trip (commit a directory, check it out elsewhere, compare bytes), a refusal to overwrite, and a diff where one file is unchanged. Then extend trees to subdirectories (mode 40000 entries) and confirm real Git still reads your objects with `git ls-tree -r`. No reference implementation or promised output is supplied.
'''


def taught_shell_series():
    stages = []
    for number, spec in enumerate(STAGES, start=1):
        checks = spec['checks']
        stages.append(dict(
            id=f'shell2-{number}', title=spec['title'], goal=spec['goal'],
            lessonRef=f'projects/shell2/{number}',
            lesson='Prerequisites: ' + spec['prerequisites'] + '\n\n' + spec['lesson'],
            language='python',
            starterFiles={'src/main.py': HEADER + dedent(spec['starter']).strip() + '\n'},
            referenceFiles={'src/main.py': HEADER + CUMULATIVE[number - 1].strip() + '\n'},
            visibleTests=checks[:1], hiddenTests=checks[1:], hints=spec['hints'],
            reviewRubric=RUBRIC, workedExample=spec['worked'], walkthrough=spec['walkthrough'],
            concepts=[spec['goal'], 'Learn freely, predict by hand, build with guidance, then check your own version.'],
            guidedPrompt='Change one input from the worked example. Predict the result by hand, then run your own code and explain any difference. Hints are free.',
            requiresHostProcess=number in (2, 3), prerequisites=spec['prerequisites']))
    return dict(
        id='shell2', title='Build your own shell and Git objects, taught (v2)',
        outline=['Hand-written tokenizer and parser', 'Processes and built-ins', 'Real pipes and redirects', 'Git-compatible object store and refs'],
        complete=True, stages=stages,
        resumeBullets=['Only after your own run: wrote a POSIX-style shell (hand-written tokenizer validated against shlex on [N] random lines, concurrent OS pipelines, redirects, built-ins).',
                       'Built a Git-compatible object store whose blob/tree/commit ids match real Git and pass `git fsck --strict` ([your verification command]).'],
        demoRecipe='Record your own asciinema session in a synthetic scratch folder: a quoted argument containing |, a three-stage pipeline, > and >>, a 127 status, two mini-git commits with a moved ref, and `git cat-file -p` reading your object. Link the cast and a transcript from the README. Stage checks 2–3 start real processes on your machine through the local CLI.',
        stretchGoals=['Job control: background &, jobs, fg, with cleanup tests.', '$VAR and $? expansion with quoting rules (expansion inside double quotes only).', 'Subdirectory trees (mode 40000) verified with `git ls-tree -r`.', 'Unsolved follow-on: FOLLOW-ON.md (checkout and diff).'],
        seniorReviewRubric=['Tokenizer: states and transitions explained; an oracle or property test exists.', 'Processes: no shell=True; statuses propagate; child processes are always waited for.', 'Pipes: the parent closes its copies; there is a test that would hang without that.', 'Object store: byte-exact format, verified hashes on read, compare-and-swap refs.', 'Security: inputs are never re-interpreted as code; scratch folders only in the demo.'],
        prerequisites=['Python strings, bytes, exceptions, classes', 'subprocess, files and paths', 'Git Lab lessons 1–4 recommended before stage 4'],
        authoringStatus='Original source fixtures, references and checks. Every reference passes its checks and every named wrong implementation is rejected by a hidden check (host test). Real-Git ids in stage 4 were produced by git on the author host. No app execution, learner outcome or mastery is implied.',
        predecessor='shell (legacy identities preserved, no progress migration)',
        supplementalFiles={'README-TEMPLATE.md': README_TEMPLATE, 'FOLLOW-ON.md': FOLLOW_ON})


MUTANTS = {f'shell2-{n}': spec['mutants'] for n, spec in enumerate(STAGES, start=1)}
