"""Original teaching and portfolio plans; no learner code is filled in."""
EXAMPLES = {
'gpt-1': ('For text "aba", sorted symbols are ["a", "b"]. The IDs are [0, 1, 0]. Decoding those IDs returns "aba", including repeated symbols.', ['Write the alphabet for a short string by hand.','Map each character to its alphabet index.','Join characters looked up by index; compare every character, including spaces.']),
'gpt-2': ('For pairs 0→1 and 1→0, add-one counts are [[1,2],[2,1]]. Each row totals 3, so the first row becomes [1/3,2/3].', ['Draw a current-token by next-token count table.','Add one to every cell, then count adjacent pairs.','Divide each cell by its own row total and check row sums.']),
'gpt-3': ('If the correct token has predicted probability 0.25, its loss is −log(0.25) ≈ 1.386. If training raises that probability to 0.5, loss drops to about 0.693.', ['Compute stable softmax on one row.','Subtract one at the target index to obtain the score gradient.','Accumulate repeated-row gradients, update once, and inspect the new loss.']),
'gpt-4': ('A one-hot input [0,1,0] selects row 1 of W1. Tanh turns that row into hidden features. Multiplying those features by W2 produces one score per possible next token.', ['Trace matrix shapes before writing operations.','Compute the forward pass and cross entropy.','Propagate gradients through W2 and tanh, then update both matrices together.']),
'gpt-5': ('With zero queries and keys, two available past positions get equal weight. Values 2 and 4 produce 3 at the second position. The first position still returns 2.', ['Sketch the lower-triangular visibility table.','Compute scaled dot products and hide future scores.','Normalize each row and multiply weights by values.']),
'gpt-6': ('For context [a,b], position 0 must produce the same features if b is replaced with c. Train only after that invariant passes; a loss decrease cannot detect future-token leakage.', ['Assemble embeddings, causal attention, residuals and normalization.','Verify causality before fitting the small NumPy readout.','Sample with a seed; compare a held-out string; optionally reveal the full PyTorch training example after your attempt.']),
'shell-1': ('echo "two words" | cat has two commands. The first receives two arguments: echo and two words. Quotes group the space; the pipe connects commands.', ['Tokenize while respecting quoted spans.','Start a new command only at an unquoted pipe.','Reject an empty command on either side of a pipe.']),
'shell-2': ('A Python process that prints 7 and exits with status 4 has stdout "7\\n" and status 4. Printing output does not mean the command succeeded.', ['Keep arguments as a list.','Run with a bounded wait and capture both output channels.','Return status, stdout and stderr without inferring success from text.']),
'shell-3': ('A producer prints "6". A consumer reads it as an integer and multiplies by 7. The pipeline returns "42" while retaining both exit statuses.', ['Run the producer and retain its status.','Use its stdout as the next process input.','Explain why buffering is simpler and why streaming large data needs a different design.']),
'shell-4': ('Two files with identical bytes can share a blob ID. Changing a filename changes the tree even when its blob stays the same. A commit also includes its parent.', ['Store typed, content-addressed blobs.','Serialize a sorted filename-to-blob tree.','Create a commit pointing to the tree and move a validated ref to that commit.']),
'agent-1': ('A fake transport receives {role: user, content: hello} and returns hi. An empty message is rejected before the fake transport is called, so no request occurs.', ['Define a request shape and a plain-text result.','Pass a fake callable into your client function.','Check validation and error behavior before considering a real provider.']),
'autopilot-1': ('Target 2 m, measured 1 m gives error 1 m. With kp=2 and other gains zero, correction is 2. At dt=0.1 s the integral grows by 0.1 m·s.', ['Label the error, timestep and gain units.','Compute proportional, integral and derivative contributions separately.','Carry state into the next deterministic simulation step and observe settling.']),
'game-1': ('Starting at x=2 with vx=4 units/s for 0.5 s gives x=4. If the arena ends at x=3, clamp the result to 3. Rendering is not part of this calculation.', ['Predict a movement step on paper.','Scale velocity by elapsed time.','Clamp both axes and check zero time and boundary crossings.']),
'contributions-1': ('A useful issue says: expected parse("a |") to reject an empty command; observed an empty process launch; reproduction is a two-line command. It describes evidence before proposing a patch.', ['Read the project license, CONTRIBUTING file and recent maintainer discussion.','Reproduce a small issue locally and add a regression test.','Draft a focused explanation; the owner decides whether to fork, post, open a PR or respond to review.']),
}
PLANS={
'gpt': ('# Live demo recipe\n\nBuild a small local web form that loads your tokenizer and model once, accepts a short prompt and bounds generated tokens. Demonstrate a seeded sample and publish the model card next to it. For an optional Hugging Face Space, review code, data license and costs first, then the owner creates and uploads it. Never bundle private training text. Record CPU latency and context limits in the demo.', ['KV cache with cached-vs-uncached output equality tests','BPE tokenizer with reversible round-trip tests','GPU scaling/loss-curve study with held-out loss, time and memory']),
'agent': ('# Live demo recipe\n\nUse the CLI with a synthetic note set and fake transport first. Record a short screencast showing retrieval citations, rejected unsafe tool use and an evaluation summary. A real provider is optional and requires explicit per-run approval and the paid cap. Blur keys and private text. Add the recording and exact reproduction command to README.', ['Grounded evaluation suite with answerable and unanswerable cases','Measured cost/latency table with model, date and sample count','Compare lexical retrieval with local embeddings on synthetic notes']),
'autopilot': ('# Live demo recipe\n\nRecord the Autopilot Lab simulation at a fixed timestep and seed. Show the same waypoint mission for a baseline and tuned controller. Export plots of height error, overshoot, settling time and actuator saturation with labeled units. Keep it a simulation demo and state sensor assumptions. The owner chooses whether to publish the recording.', ['LQR versus PID on identical disturbances','MPC comparison with compute-budget measurements','Kalman sensor-fusion error under controlled dropout']),
'shell': ('# Live demo recipe\n\nRecord a short asciinema session on macOS or a terminal video on Windows. Show quoted parsing, a pipeline, an error, two mini-git commits and a moved ref. Keep commands in a synthetic scratch repo. Put the cast/video and a transcript in docs/demo, then link them from README. The owner uploads or embeds the recording.', ['Foreground/background job control with cleanup tests','Concurrent streaming pipelines with backpressure tests','Tree diff and history traversal in mini-git']),
'game': ('# Live demo recipe\n\nPackage a playable LÖVE build with controls, a restart path and a deterministic seed. For GitHub Pages, the owner can choose a separately reviewed web-compatible build; test asset licenses and keyboard/touch input first. Show a short match against each opponent and report win rate with the seed set and number of games. The Python port is a separate runnable release.', ['A* versus behavior-tree opponent on fixed maps','RL policy evaluated against an unchanged baseline','Replay files with seed and input sequence']),
'contributions': ('# Contribution demo recipe\n\nShow a minimal reproduction, the regression test before/after your patch and a focused diff. Draft a PR description and a courteous response to review locally. Keep issue status, authorship and maintainer feedback factual. Never represent an unsubmitted draft as an accepted contribution.', ['Review another small patch constructively','Document a benchmark without overstating its scope','Propose a narrow Morrowdock open-core improvement after publication/license approval']),
}

def augment(series):
    for p in series:
        p['demoRecipe'],p['stretchGoals']=PLANS[p['id']]
        p['seniorReviewRubric']=[
            'Code quality: explain invariants, errors, ownership and module boundaries.',
            'Tests: demonstrate a caught regression, edge cases and coverage gaps.',
            'Documentation: reproduce quickstart and demo in a clean checkout.',
            'Naming: make units, shapes and side effects explicit.',
            'Security: inspect inputs, subprocesses, secrets, licenses and private data.'
        ]
        p['resumeBullets']=[f"Built {p['title'].lower().replace('build your own ', '').replace('build a ', '')}; measured [measured quality metric] against [measured baseline] on [measured evaluation set].",'Reduced [measured runtime or error] from [measured baseline] to [measured result], with reproducible tests and documented limitations.']
        for s in p['stages']:
            s['workedExample'],s['walkthrough']=EXAMPLES[s['id']]
            s['concepts']=[s['goal'], 'Predict an output, trace each step, then explain one boundary case.']
            s['requiresHostProcess'] = s['id'] in ['shell-2', 'shell-3']
            s['guidedPrompt']='Change one input in the worked example. Predict the result, run an example, then explain any difference. Hints are free.'
            s['starterFiles']={'src/'+n:code for n,code in s['starterFiles'].items()}
            s['referenceFiles']={'src/'+n:code for n,code in s['referenceFiles'].items()}

# Explicit starter annotations help the learner reason about interfaces before implementation.
SIGNATURES = {
'gpt-1': {'vocabulary':'text: str) -> list[str]', 'encode':'text: str, vocab: list[str]) -> list[int]', 'decode':'ids: list[int], vocab: list[str]) -> str'},
'gpt-2': {'bigram':'ids: list[int], size: int) -> np.ndarray'},
'gpt-3': {'train':'ids: list[int], size: int, steps: int = 80, lr: float = 0.5) -> tuple[np.ndarray, list[float]]'},
'gpt-4': {'train_mlp':'x: np.ndarray, y: list[int], classes: int, steps: int = 250, lr: float = 0.3) -> tuple[np.ndarray, np.ndarray, list[float]]'},
'gpt-5': {'attention':'q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]'},
'gpt-6': {'__init__':'self, size: int, width: int = 8, context: int = 16, seed: int = 3) -> None','features':'self, ids: list[int]) -> np.ndarray','logits':'self, ids: list[int]) -> np.ndarray','fit':'self, ids: list[int], steps: int = 120, lr: float = 0.3) -> list[float]','generate':'self, ids: list[int], count: int, seed: int = 1) -> list[int]'},
'shell-1': {'parse':'line: str) -> list[list[str]]'},
'shell-2': {'execute':'argv: list[str]) -> tuple[int, str, str]'},
'shell-3': {'pipeline':"commands: list[list[str]], text: str = '') -> tuple[list[int], str]"},
'shell-4': {'__init__':'self, root: str | Path) -> None','put':'self, kind: str, data: bytes) -> str','commit':'self, files: dict[str, bytes], parent: str | None = None) -> str','read':'self, oid: str) -> list[bytes]','set_ref':'self, name: str, oid: str) -> None','get_ref':'self, name: str) -> str'},
'agent-1': {'ask':'message: str, transport: Callable[[dict[str, str]], str]) -> str'},
'autopilot-1': {'pid':'target: float, measured: float, integral: float, previous_error: float, dt: float, kp: float, ki: float, kd: float) -> tuple[float, float, float]'},
'contributions-1': {'issue_draft':'title: str, expected: str, observed: str, steps: list[str]) -> dict[str, str | list[str]]'},
}
_base_augment=augment
def augment(series):
    import re
    _base_augment(series)
    for p in series:
        for stage in p['stages']:
            for field in ['starterFiles','referenceFiles']:
                for name, code in stage[field].items():
                    for fn, sig in SIGNATURES.get(stage['id'],{}).items():
                        code=re.sub(r'(?m)^(\s*)def '+fn+r'\([^\n]*?\):',lambda m:m.group(1)+'def '+fn+'('+sig+':',code)
                    if stage['id']=='agent-1': code='from typing import Callable\n'+code
                    stage[field][name]=code
