"""Taught 'Build your own GPT' (series gpt2): original lessons, checks and wrong implementations.

Additive: the legacy `gpt` series and its six stage identities stay byte-identical; no receipt
migrates. Host authoring data only. Nothing here executes learner code or admits a runtime.
"""
from textwrap import dedent

from gpt_taught_refs import CUMULATIVE

HEADER = 'import math\nimport numpy as np\n'
PRELUDE = 'import math\nimport numpy as np\n'


def check(name, source):
    return dict(name=name, source=PRELUDE + dedent(source).strip() + '\n')


RUBRIC = [
    'Predict a result by hand before running, then explain any difference.',
    'Show the boundary case your own test covers and why it discriminates.',
    'Explain one bug you hit, the counterexample that exposed it and the fix.',
    'Describe help used neutrally; authored examples are not your measured results.',
]

STAGES = [
    dict(
        title='1 · Tokens from merges (byte-pair encoding)',
        goal='Learn BPE merges from training text, encode new text with them in order, and decode losslessly.',
        prerequisites='Python lists, dicts, tuples and while-loops. Recommended first: tinylm2 stage 1 (training-only vocabulary).',
        lesson='''A model never sees letters, only token ids. Characters give a tiny vocabulary but long sequences; whole words give short sequences but an endless vocabulary, and any new word is unknown. Byte-pair encoding (BPE) sits between them. Start from characters, then repeatedly find the most frequent adjacent pair and glue it into one new symbol. Each glue step is a merge, and the ordered list of merges is the tokenizer.

Three rules make it deterministic and testable:
1. Count every adjacent pair, overlapping ones included: “aaaa” has the pair (a,a) three times.
2. When merging, scan left to right and never reuse a symbol: “aaaa” becomes [aa, aa], not [aa, a, a] or [a, aa, a].
3. Ties go to the smallest pair in Python's tuple order, so two runs always agree. Stop early when no pair occurs at least twice, because a merge seen once only memorises.

Encoding new text replays the learned merges in the order they were learned, not by frequency in the new text. Decoding is joining the strings, and because every merge only concatenates, decode(encode(x)) == x for any text, even characters never seen in training. Those stay single-character tokens. That losslessness is BPE's advantage over an <UNK> token.

Misconception: “the merge count is the vocabulary size.” Not quite: the vocabulary is the training characters plus one new symbol per merge, and you may get fewer merges than you asked for.
Counterexample: in “aaaa” the pair count (3) is not the number of tokens a merge produces (2). If you test only on text without repeated letters, an overlapping-merge bug never shows.

Real tokenizers such as GPT-2's run BPE on bytes and split on spaces first. This stage keeps spaces as ordinary symbols, so merges can cross word boundaries. Say so in your README.

Guided activity A: by hand, learn 4 merges from “low lower lowest”, then encode “slow blow”. Guided activity B: predict train_bpe(“aaaa”, 3) and encode “aaaaa” with it before running.
On your own: the hidden checks use different text, a tie and Unicode. Write your own test of the tie-break before you look.''',
        starter='''# Stage 1. Write these yourself; the tests call them by name.

def pair_counts(symbols: list[str]) -> dict[tuple[str, str], int]:
    raise NotImplementedError

def merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    raise NotImplementedError

def train_bpe(text: str, merges: int) -> list[tuple[str, str]]:
    raise NotImplementedError

def encode_bpe(text: str, merges: list[tuple[str, str]]) -> list[str]:
    raise NotImplementedError

def decode_bpe(tokens: list[str]) -> str:
    raise NotImplementedError''',
        worked='''“low lower lowest”. Pair counts: (l,o)=3, (o,w)=3, ( ,l)=2, (w,e)=2, (w, )=1, … Tie between (l,o) and (o,w); the smaller tuple is (l,o), so merge 1 = (l,o). Then (lo,w)=3, so merge 2 = (lo,w). Now “low” appears three times and the most frequent pairs are ( ,low) and (low,e) with 2 each; ( ,low) is smaller (space sorts before letters), so merge 3 = ( ,low). Then ( low,e)=2 gives merge 4. Encoded: [low, ' lowe', r, ' lowe', s, t]. Encoding “slow blow” with the same merges gives [s, low, ' ', b, low]: the leading-space merge needs a space right before “low”, which “slow” doesn't have.''',
        walkthrough=['Count overlapping adjacent pairs with zip(symbols, symbols[1:]).',
                     'Pick the highest count; break ties with min() over the tied pairs; stop if the best count is below 2.',
                     'Merge left to right, skipping two positions after each merge.',
                     'Encode by replaying merges in learned order; decode by joining.'],
        hints=['Write merge_pair first and test it on ["a","a","a","a"] with ("a","a").',
               'Ties: collect every pair with the best count, then take min() of them.'],
        checks=[
            check('worked example merges and encoding', '''
assert pair_counts(list('aaaa')) == {('a','a'): 3}
m = train_bpe('low lower lowest', 4)
assert m == [('l','o'), ('lo','w'), (' ','low'), (' low','e')], m
assert encode_bpe('low lower lowest', m) == ['low',' lowe','r',' lowe','s','t']
assert encode_bpe('slow blow', m) == ['s','low',' ','b','low']
'''),
            check('overlapping pairs merge without reuse, and early stop', '''
m = train_bpe('aaaa', 3)
assert m == [('a','a')], 'one merge: afterwards (aa,aa) occurs once only'
assert encode_bpe('aaaaa', m) == ['aa','aa','a']
assert merge_pair(['a','a','a'], ('a','a')) == ['aa','a']
assert train_bpe('abcd', 5) == []
assert train_bpe('', 2) == [] and encode_bpe('', [('a','b')]) == []
'''),
            check('ties use tuple order and merges replay in learned order', '''
assert train_bpe('xyab xyab', 1) == [('a','b')], 'three pairs tie at 2; the smallest tuple wins'
assert train_bpe('bcbcadad', 1) == [('a','d')]
merges = [('c','a'), ('a','t')]
assert encode_bpe('cat at at', merges) == ['ca','t',' ','at',' ','at'], 'replay merges in learned order, not by frequency in the new text'
'''),
            check('lossless on unseen text and invalid merge counts', '''
m = train_bpe('the cat sat on the mat', 6)
for text in ['the hat', 'zebra ü 🙂', 'thethe', ' ', 'tt']:
    tokens = encode_bpe(text, m)
    assert decode_bpe(tokens) == text and all(isinstance(t, str) and t for t in tokens)
for bad in [-1, 1.5]:
    try: train_bpe('abab', bad)
    except ValueError: pass
    else: raise AssertionError('reject a merge count that is not a non-negative whole number')
'''),
        ],
        mutants=[
            ('frequency-order encode', 'encode_bpe', '''
def encode_bpe(text, merges):
    symbols = list(text)
    for pair in sorted(merges, key=lambda p: -pair_counts(symbols).get(p, 0)):
        symbols = merge_pair(symbols, pair)
    return symbols
'''),
            ('merge reuses symbols', 'merge_pair', '''
def merge_pair(symbols, pair):
    out = []
    for i, s in enumerate(symbols):
        if i + 1 < len(symbols) and (s, symbols[i + 1]) == pair: out.append(s + symbols[i + 1])
        elif i > 0 and (symbols[i - 1], s) == pair: continue
        else: out.append(s)
    return out
'''),
            ('ties by first seen', 'train_bpe', '''
def train_bpe(text, merges):
    if not isinstance(merges, int) or merges < 0: raise ValueError('bad')
    symbols, learned = list(text), []
    for _ in range(merges):
        counts = pair_counts(symbols)
        if not counts or max(counts.values()) < 2: break
        pair = max(counts, key=counts.get)
        learned.append(pair); symbols = merge_pair(symbols, pair)
    return learned
'''),
            ('no early stop', 'train_bpe', '''
def train_bpe(text, merges):
    if not isinstance(merges, int) or merges < 0: raise ValueError('bad')
    symbols, learned = list(text), []
    for _ in range(merges):
        counts = pair_counts(symbols)
        if not counts: break
        best = max(counts.values()); pair = min(p for p, c in counts.items() if c == best)
        learned.append(pair); symbols = merge_pair(symbols, pair)
    return learned
'''),
        ]),
    dict(
        title='2 · Scores, softmax and loss',
        goal='Turn raw scores (logits) into probabilities stably and measure them with cross-entropy.',
        prerequisites='Stage 1; NumPy arrays and axis=; exp and natural log. Recommended: tinylm2 stage 3 (held-out loss).',
        lesson='''A neural language model doesn't output probabilities directly. For each position it outputs one raw score per vocabulary token, called logits. Softmax turns a row of logits z into probabilities: p_i = exp(z_i) / Σ_j exp(z_j). Bigger score, bigger share; every share is positive; the row sums to 1.

Two facts you will use constantly:
1. Softmax ignores adding the same constant to every logit in a row. So subtract the row's maximum before exp. The result is identical, and exp never sees a huge number. Without this, logits like [1000, 999] overflow to inf/inf = nan.
2. Cross-entropy is the average of −ln p(target) over positions, measured in nats. A model that knows nothing (all logits equal) scores ln V. With V=3 that is ln 3 ≈ 1.0986. This is your first baseline: any trained model should beat ln V on held-out text, or it has learned nothing.

Compute cross-entropy straight from logits with log-softmax: log p = (z − max) − ln Σ exp(z − max). Computing softmax and then log is mathematically the same but loses precision when a probability underflows to 0.

Misconception: “softmax along any axis is fine.” Rows are positions and columns are vocabulary entries. Normalising down a column mixes different positions' predictions.
Counterexample: [[2,1,0],[0,0,5]] normalised by column gives a first row that doesn't sum to 1.

Guided activity A: by hand, softmax [2,1,0] ≈ [0.665, 0.245, 0.090]; the loss for target 0 is −ln 0.665 ≈ 0.408, and for target 2 it is ≈ 2.408. Guided activity B: predict what happens to softmax([1000, 999]) in a naive implementation, then in yours.
On your own: hidden checks use big logits, several rows and bad targets.''',
        starter='''# Keep your stage-1 functions above. Stage 2:
import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    raise NotImplementedError

def cross_entropy(logits: np.ndarray, targets: list[int]) -> float:
    raise NotImplementedError''',
        worked='''Logits [2,1,0]. Subtract the max: [0,−1,−2]. exp gives [1, 0.3679, 0.1353], sum 1.5032, so p ≈ [0.6652, 0.2447, 0.0900]. For target 0, the loss is −ln 0.6652 = 0.4076 nats. For target 2, it is 0.4076 + 2 = 2.4076: the log-probability of entry i is z_i − max − ln Σ, so the targets' losses differ by exactly their logit gap. Equal logits over V=3 give ln 3 = 1.0986.''',
        walkthrough=['Shift by the row max with keepdims=True so the shapes broadcast.',
                     'Exponentiate and divide by the row sum.',
                     'For the loss, build log-softmax, pick each row\'s target entry, average and negate.',
                     'Validate shape (one target per row) and target range before computing.'],
        hints=['axis=-1 means "across the vocabulary" for both 1-D and 2-D inputs.',
               'np.arange(n) pairs with the target array to pick one entry per row.'],
        checks=[
            check('worked example and uniform baseline', '''
p = softmax(np.array([[2.,1.,0.]]))
assert np.allclose(p, [[0.66524096, 0.24472847, 0.09003057]], atol=1e-8)
assert math.isclose(cross_entropy(np.array([[2.,1.,0.]]), [0]), 0.4076059644443804, abs_tol=1e-12)
assert math.isclose(cross_entropy(np.zeros((4,3)), [0,1,2,1]), math.log(3), abs_tol=1e-12)
'''),
            check('stable for huge logits and shift invariant', '''
p = softmax(np.array([[1000., 999.], [-1000., -1000.]]))
assert np.isfinite(p).all() and np.allclose(p[0], [1/(1+math.exp(-1)), 1/(1+math.exp(1))]) and np.allclose(p[1], .5)
z = np.array([[3.,-1.,2.],[0.,5.,1.]])
assert np.allclose(softmax(z), softmax(z + 7.5))
loss = cross_entropy(np.array([[0., 800.]]), [0])
assert math.isclose(loss, 800., rel_tol=1e-12), 'log-softmax keeps a finite, exact large loss'
'''),
            check('rows, not columns, and a mean over positions', '''
z = np.array([[2.,1.,0.],[0.,0.,5.]])
p = softmax(z)
assert np.allclose(p.sum(axis=1), 1) and p[1].argmax() == 2
expected = (-math.log(p[0,1]) - math.log(p[1,2])) / 2
assert math.isclose(cross_entropy(z, [1,2]), expected, abs_tol=1e-12)
assert np.allclose(softmax(np.array([1.,2.,3.])).sum(), 1)
'''),
            check('rejects mismatched or out-of-range targets', '''
for logits, targets in [(np.zeros((2,3)), [0]), (np.zeros((2,3)), [0,3]), (np.zeros((1,3)), [-1]), (np.zeros((0,3)), [])]:
    try: cross_entropy(logits, targets)
    except (ValueError, IndexError): pass
    else: raise AssertionError(f'reject {targets} for shape {logits.shape}')
'''),
        ],
        mutants=[
            ('no max shift', 'softmax', '''
def softmax(logits):
    e = np.exp(np.asarray(logits, dtype=float)); return e / e.sum(axis=-1, keepdims=True)
'''),
            ('column normalisation', 'softmax', '''
def softmax(logits):
    z = np.asarray(logits, dtype=float); z = z - z.max(); e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)
'''),
            ('sum instead of mean', 'cross_entropy', '''
def cross_entropy(logits, targets):
    z = np.asarray(logits, dtype=float); t = np.asarray(targets)
    if z.ndim != 2 or t.shape != (z.shape[0],) or len(t) == 0 or (t < 0).any() or (t >= z.shape[1]).any(): raise ValueError('bad')
    return float(-np.log(softmax(z)[np.arange(len(t)), t]).sum())
'''),
            ('log of softmax underflows', 'cross_entropy', '''
def cross_entropy(logits, targets):
    z = np.asarray(logits, dtype=float); t = np.asarray(targets)
    if z.ndim != 2 or t.shape != (z.shape[0],) or len(t) == 0 or (t < 0).any() or (t >= z.shape[1]).any(): raise ValueError('bad')
    with np.errstate(divide='ignore'): return float(-np.log(softmax(z)[np.arange(len(t)), t]).mean())
'''),
        ]),
    dict(
        title='3 · Learn instead of count (gradient descent)',
        goal='Derive and check the softmax cross-entropy gradient, and train a bigram logit table by gradient descent.',
        prerequisites='Stage 2; derivatives of exp and ln; the chain rule; matrix indexing.',
        lesson='''In tinylm2 you counted transitions. Now the same model is learned the way every neural network is: a table W of logits (row = current token, column = next token), a loss, and repeated small steps downhill.

The one derivative to know: for one position with probabilities p = softmax(z) and target y, ∂loss/∂z = p − onehot(y). Derivation: loss = −z_y + ln Σ exp(z_j). The derivative of ln Σ exp(z_j) with respect to z_i is p_i, and −z_y contributes −1 at i = y. For a mean over N positions, divide by N.
Each position's gradient lands on the row of W for that position's current token, so rows used several times add up. That needs np.add.at, not plain assignment.

Before trusting any hand-written gradient, check it numerically: nudge one entry by ±ε, recompute the loss, and compare (L(w+ε) − L(w−ε)) / 2ε with your gradient. This finite-difference check is how you catch a missing /N or a wrong sign.

What training discovers: with zero initial weights and enough steps, each observed row's softmax approaches that row's transition frequencies, the same answer as counting with k=0. A row whose token never appears as a current token gets zero gradient forever, so it stays exactly uniform. Smoothing in counts and initialisation in learning both decide what unseen contexts look like.

Misconception: “lower training loss always means a better model.” This model can only reach the training frequencies, and on held-out text an over-confident row can lose to uniform (see tinylm2's counterexample).

Guided activity A: for ids [0,1,0,1,0,2] and W = 0, write the gradient by hand. Row 0 is used three times with targets 1,1,2: (3·(1/3) − [0,2,1]) / 5 = [0.2, −0.2, 0]. Guided activity B: train 300 steps at lr=1 and compare row 0 with the counts 2/3 and 1/3.
On your own: the hidden checks run a finite-difference check on random weights.''',
        starter='''# Keep stages 1–2 above. Stage 3:

def bigram_loss_and_grad(w: np.ndarray, ids: list[int]) -> tuple[float, np.ndarray]:
    raise NotImplementedError

def train_bigram(ids: list[int], size: int, steps: int = 300, lr: float = 1.0) -> tuple[np.ndarray, list[float]]:
    # Start from zeros; return the final weights and the loss before every step plus the final loss.
    raise NotImplementedError''',
        worked='''ids [0,1,0,1,0,2] have 5 transitions: 0→1, 1→0, 0→1, 1→0, 0→2. With W = 0 every p is 1/3 and the loss is ln 3 = 1.0986. Gradient row 0 (used 3×, targets 1,1,2): [1−0, 1−2, 1−1]/5 = [0.2, −0.2, 0]. Row 1 (used 2×, target 0 both times): [2/3−2, 2/3, 2/3]/5 = [−0.2667, 0.1333, 0.1333]. Row 2 is never a current token: gradient 0. After 300 steps at lr=1, row 0 ≈ [0.004, 0.665, 0.331] (counts say 0, 2/3, 1/3), row 1 ≈ [0.994, 0.003, 0.003], row 2 stays [1/3,1/3,1/3]. The loss falls from 1.0986 to ≈ 0.386.''',
        walkthrough=['Split ids into current tokens x = ids[:-1] and next tokens y = ids[1:].',
                     'Forward: logits = w[x]; loss = cross_entropy(logits, y).',
                     'Backward: d = softmax(logits); subtract 1 at each target; divide by N; scatter-add rows into a zero gradient.',
                     'Train: start at zeros, record the loss, step w -= lr * grad.'],
        hints=['np.add.at(grad, x, d) adds repeated rows; grad[x] += d silently keeps only one of them.',
               'Finite-difference check: perturb one entry, recompute the loss, compare.'],
        checks=[
            check('worked example gradient and uniform start', '''
loss, g = bigram_loss_and_grad(np.zeros((3,3)), [0,1,0,1,0,2])
assert math.isclose(loss, math.log(3), abs_tol=1e-12)
assert np.allclose(g, [[.2,-.2,0],[-4/15,2/15,2/15],[0,0,0]], atol=1e-12)
'''),
            check('finite-difference gradient check on random weights', '''
rng = np.random.default_rng(11)
w = rng.normal(0, 1, (4,4)); ids = [3,1,1,0,2,1,3,3,0]
loss, g = bigram_loss_and_grad(w, ids)
eps = 1e-6
for i in range(4):
    for j in range(4):
        wp = w.copy(); wp[i,j] += eps; wm = w.copy(); wm[i,j] -= eps
        numeric = (bigram_loss_and_grad(wp, ids)[0] - bigram_loss_and_grad(wm, ids)[0]) / (2*eps)
        assert abs(numeric - g[i,j]) < 1e-6, (i, j, numeric, g[i,j])
'''),
            check('training approaches counts; unseen row stays uniform', '''
w, losses = train_bigram([0,1,0,1,0,2], 3, 300, 1.0)
assert len(losses) == 301 and losses[-1] < losses[0] and all(b <= a + 1e-12 for a, b in zip(losses, losses[1:]))
p = softmax(w)
assert np.allclose(p[0], [0, 2/3, 1/3], atol=0.02) and np.allclose(p[1], [1, 0, 0], atol=0.02)
assert np.allclose(p[2], 1/3, atol=1e-12)
'''),
            check('different data and a single transition', '''
w, losses = train_bigram([2,0,2,0,2,1,1], 3, 400, 1.0)
p = softmax(w)
assert np.allclose(p[2], [2/3, 1/3, 0], atol=0.03) and np.allclose(p[0], [0, 0, 1], atol=0.03)
loss, g = bigram_loss_and_grad(np.zeros((2,2)), [1,0])
assert np.allclose(g, [[0,0],[-.5,.5]])
try: bigram_loss_and_grad(np.zeros((2,2)), [1])
except (ValueError, IndexError): pass
else: raise AssertionError('a single token has no transition to learn')
'''),
        ],
        mutants=[
            ('missing division by N', 'bigram_loss_and_grad', '''
def bigram_loss_and_grad(w, ids):
    x, y = np.asarray(ids[:-1]), np.asarray(ids[1:])
    if len(y) == 0: raise ValueError('bad')
    p = softmax(w[x]); loss = cross_entropy(w[x], y.tolist()); d = p.copy(); d[np.arange(len(y)), y] -= 1
    g = np.zeros_like(w, dtype=float); np.add.at(g, x, d); return loss, g
'''),
            ('assignment drops repeated rows', 'bigram_loss_and_grad', '''
def bigram_loss_and_grad(w, ids):
    x, y = np.asarray(ids[:-1]), np.asarray(ids[1:])
    if len(y) == 0: raise ValueError('bad')
    p = softmax(w[x]); loss = cross_entropy(w[x], y.tolist()); d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    g = np.zeros_like(w, dtype=float); g[x] += d; return loss, g
'''),
            ('sign flipped', 'bigram_loss_and_grad', '''
def bigram_loss_and_grad(w, ids):
    x, y = np.asarray(ids[:-1]), np.asarray(ids[1:])
    if len(y) == 0: raise ValueError('bad')
    p = softmax(w[x]); loss = cross_entropy(w[x], y.tolist()); d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    g = np.zeros_like(w, dtype=float); np.add.at(g, x, -d); return loss, g
'''),
            ('next and current swapped', 'bigram_loss_and_grad', '''
def bigram_loss_and_grad(w, ids):
    y, x = np.asarray(ids[:-1]), np.asarray(ids[1:])
    if len(y) == 0: raise ValueError('bad')
    p = softmax(w[x]); loss = cross_entropy(w[x], y.tolist()); d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    g = np.zeros_like(w, dtype=float); np.add.at(g, x, d); return loss, g
'''),
        ]),
    dict(
        title='4 · More context: embeddings and a hidden layer',
        goal='Predict from the previous two tokens with learned embeddings and a tanh hidden layer, verified by gradient checks.',
        prerequisites='Stage 3; matrix multiplication shapes; tanh and its derivative 1 − tanh².',
        lesson='''A bigram sees one token. In “1 2 3 4 2 5” repeated, the token after 2 is 3 if the token before was 1, and 5 if it was 4. No bigram can know which. Its best held-out loss stays stuck above zero, because both transitions out of 2 look the same to it.

Give the model a context of the previous `block` tokens:
1. make_contexts pads the start with token 0 and returns, for each position, the block tokens before it and the target at it. The target is never part of its own context. Including it is the classic off-by-one that makes training look perfect.
2. Embeddings: each token id indexes a learned row of E (V × d). The context's rows are concatenated into one vector of length block·d.
3. Hidden layer: h = tanh(flat·W1 + b1). Logits: h·W2 + b2. The loss is your stage-2 cross-entropy.

Backpropagation is the chain rule applied in reverse, and the shapes tell you each product:
d_logits = (softmax − onehot)/N; dW2 = hᵀ·d_logits; db2 = Σ rows; dh = d_logits·W2ᵀ; dpre = dh ⊙ (1 − h²); dW1 = flatᵀ·dpre; db1 = Σ rows; d_flat = dpre·W1ᵀ. Reshape d_flat back to (N, block, d) and scatter-add it into dE with np.add.at, because a token can appear in many contexts, and twice in one context.

Gradient-check every parameter with finite differences before training. The two classic bugs are using 1 − h instead of 1 − h², and losing repeated embedding gradients; both show up immediately.

Misconception: “a bigger model is always better.” A block-1 MLP on this data is no better than a bigram: dev loss ≈ 0.23 vs 0.26. Only the extra context helps (block 2: ≈ 0.003). Capacity without the right information doesn't help.

Guided activity A: write make_contexts([5,6,7], 2) by hand: [[0,0],[0,5],[5,6]] with targets [5,6,7]. Guided activity B: train block 1 and block 2 on the pattern, then explain the dev losses.
On your own: the hidden checks gradient-check a context where one token repeats, and use a different ambiguous pattern.''',
        starter='''# Keep stages 1–3 above. Stage 4:

def make_contexts(ids: list[int], block: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError

def init_mlp(size: int, block: int, embed: int = 4, hidden: int = 16, seed: int = 0) -> dict:
    # Return {'E','W1','b1','W2','b2'} with a seeded np.random.default_rng: E normal(0, .5); W1 normal(0, 1/sqrt(block*embed)); W2 normal(0, 1/sqrt(hidden)); biases zero. (Very small inits, e.g. 0.01, train too slowly for the checks.)
    raise NotImplementedError

def mlp_forward(params: dict, contexts: np.ndarray) -> tuple[np.ndarray, dict]:
    raise NotImplementedError

def mlp_loss_and_grads(params: dict, contexts: np.ndarray, targets: np.ndarray) -> tuple[float, dict]:
    raise NotImplementedError

def train_mlp(params: dict, contexts: np.ndarray, targets: np.ndarray, steps: int = 400, lr: float = 0.5) -> list[float]:
    raise NotImplementedError''',
        worked='''Pattern [1,2,3,4,2,5]: train = 8 repetitions, dev = 3 repetitions, vocabulary 6 (0 is padding). Results with the reference: a bigram trained 400 steps reaches a dev loss of ≈ 0.257. A block-1 MLP reaches ≈ 0.233, about the same, because it has the same information. A block-2 MLP (embed 4, hidden 16, seed 0, 400 steps, lr 0.5) goes from a training loss of ≈ 1.78 to ≈ 0.002, with a dev loss of ≈ 0.003. The dev text repeats the training pattern, so this demonstrates the context effect, not generalisation to new text.''',
        walkthrough=['Write make_contexts and check it on [5,6,7] by hand.',
                     'Forward: index E, reshape to (N, block·d), tanh layer, logits.',
                     'Backward in reverse order; keep each gradient the same shape as its parameter.',
                     'Gradient-check all five parameters, then train and compare with a bigram on dev.'],
        hints=['params["E"][contexts] has shape (N, block, d); .reshape(N, -1) concatenates.',
               'np.add.at(grads["E"], contexts, d_emb) handles repeats inside one context too.'],
        checks=[
            check('contexts pad the start and exclude the target', '''
c, y = make_contexts([5,6,7], 2)
assert c.tolist() == [[0,0],[0,5],[5,6]] and y.tolist() == [5,6,7]
c, y = make_contexts([4,4], 3)
assert c.tolist() == [[0,0,0],[0,0,4]] and y.tolist() == [4,4]
'''),
            check('gradient check on every parameter (repeated tokens)', '''
p = init_mlp(5, 3, embed=3, hidden=4, seed=2)
c = np.array([[1,1,2],[3,1,1],[0,4,4]]); t = np.array([2,0,1])
loss, g = mlp_loss_and_grads(p, c, t)
eps = 1e-6
for name in ['E','W1','b1','W2','b2']:
    assert g[name].shape == p[name].shape, name
    flat = p[name].reshape(-1)
    for i in range(0, flat.size, max(1, flat.size // 12)):
        old = flat[i]
        flat[i] = old + eps; up = mlp_loss_and_grads(p, c, t)[0]
        flat[i] = old - eps; down = mlp_loss_and_grads(p, c, t)[0]
        flat[i] = old
        assert abs((up - down) / (2*eps) - g[name].reshape(-1)[i]) < 1e-6, (name, i)
'''),
            check('two tokens of context beat the bigram on held-out text', '''
pat = [1,2,3,4,2,5]; train, dev = pat*8, pat*3
c, y = make_contexts(train, 2); p = init_mlp(6, 2, seed=0); losses = train_mlp(p, c, y, 400, 0.5)
assert losses[-1] < losses[0]
cd, yd = make_contexts(dev, 2); mlp_dev = cross_entropy(mlp_forward(p, cd)[0], list(yd))
w, _ = train_bigram(train, 6, 400, 1.0); bigram_dev = bigram_loss_and_grad(w, dev)[0]
assert mlp_dev < 0.05 and bigram_dev > 0.2, (mlp_dev, bigram_dev)
'''),
            check('a different ambiguous pattern and determinism', '''
c1, y1 = make_contexts([7,8,9], 1)
assert c1.tolist() == [[0],[7],[8]] and y1.tolist() == [7,8,9], 'each context ends just before its target'
pat = [3,1,2,4,1,5,2]; c, y = make_contexts(pat*6, 2)
a = init_mlp(6, 2, seed=4); b = init_mlp(6, 2, seed=4)
assert all(np.array_equal(a[k], b[k]) for k in a)
la = train_mlp(a, c, y, 300, 0.5); lb = train_mlp(b, c, y, 300, 0.5)
assert la == lb and la[-1] < 0.1
'''),
        ],
        mutants=[
            ('tanh derivative 1 - h', 'mlp_loss_and_grads', '''
def mlp_loss_and_grads(params, contexts, targets):
    logits, cache = mlp_forward(params, contexts); loss = cross_entropy(logits, list(targets))
    d = softmax(logits); d[np.arange(len(targets)), targets] -= 1; d /= len(targets)
    g = {'W2': cache['h'].T @ d, 'b2': d.sum(0)}; dh = (d @ params['W2'].T) * (1 - cache['h'])
    g['W1'] = cache['flat'].T @ dh; g['b1'] = dh.sum(0); de = (dh @ params['W1'].T).reshape(*contexts.shape, -1)
    g['E'] = np.zeros_like(params['E']); np.add.at(g['E'], contexts, de); return loss, g
'''),
            ('embedding repeats overwritten', 'mlp_loss_and_grads', '''
def mlp_loss_and_grads(params, contexts, targets):
    logits, cache = mlp_forward(params, contexts); loss = cross_entropy(logits, list(targets))
    d = softmax(logits); d[np.arange(len(targets)), targets] -= 1; d /= len(targets)
    g = {'W2': cache['h'].T @ d, 'b2': d.sum(0)}; dh = (d @ params['W2'].T) * (1 - cache['h'] ** 2)
    g['W1'] = cache['flat'].T @ dh; g['b1'] = dh.sum(0); de = (dh @ params['W1'].T).reshape(*contexts.shape, -1)
    g['E'] = np.zeros_like(params['E']); g['E'][contexts] += de; return loss, g
'''),
            ('target leaks into context', 'make_contexts', '''
def make_contexts(ids, block):
    if block < 1: raise ValueError('bad')
    padded = [0] * (block - 1) + list(ids)
    return np.array([padded[i:i + block] for i in range(len(ids))], dtype=int).reshape(len(ids), block), np.array(ids, dtype=int)
'''),
        ]),
    dict(
        title='5 · Look back with causal self-attention',
        goal='Implement single-head scaled dot-product attention with a causal mask, and verify what it can and cannot see.',
        prerequisites='Stages 2–4; dot products; matrix transpose; the stage-2 softmax.',
        lesson='''The MLP has a fixed window and a separate weight for every slot in it. Attention lets every position decide how much to read from each earlier position, based on content.

For a sequence x (n × d), project three ways: queries Q = xWq (what am I looking for?), keys K = xWk (what do I offer?) and values V = xWv (what do I pass on?). Position i scores position j by Q_i·K_j / √d_k. Dividing by √d_k keeps scores from growing with the width, which would push softmax towards a single one-hot spike and tiny gradients. Each row of scores goes through your softmax, and the output is weights·V: a weighted average of earlier values.

A language model must not peek at the future when predicting the next token. So before the softmax, set every score where j > i to −∞. Those weights become exactly 0, and each row still sums to 1. Masking after the softmax (multiplying by 0) is wrong: rows no longer sum to 1, so outputs shrink for early positions.

Two consequences you can test:
1. Causality: changing token 2 must not change outputs 0 or 1.
2. No order inside the past: attention alone treats the earlier tokens as a set. Swapping x₀ and x₁ leaves output 2 unchanged. This is why stage 6 adds position embeddings.

Misconception: “attention weights explain the model.” They show where a position read from in this layer, not why the prediction came out as it did. Treat them as a diagnostic, not an explanation.

Guided activity A: with Wq = Wk = Wv = I and x = [[1,0],[0,1],[1,1]], compute row 1 by hand. The scores are [0, 1]/√2, so the weights are 1 : e^(0.7071) = [0.3302, 0.6698]. Guided activity B: remove the mask and predict which outputs change.
On your own: the hidden checks test the causality and order properties on random inputs.''',
        starter='''# Keep stages 1–4 above. Stage 5:

def causal_attention(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Return (outputs n×d_v, weights n×n).
    raise NotImplementedError''',
        worked='''x = [[1,0],[0,1],[1,1]], with all three projections equal to the identity, so Q = K = V = x and d_k = 2. Row 0 can only see itself: weight [1,0,0], output [1,0]. Row 1 scores [0,1]/√2 = [0, 0.7071], so the weights are [0.3302, 0.6698, 0] and the output is [0.3302, 0.6698]. Row 2 scores [1,1,2]/√2, so the weights are [0.2483, 0.2483, 0.5035] and the output is [0.7517, 0.7517]. Without the √2, row 1 would be [0.2689, 0.7311]: sharper.''',
        walkthrough=['Compute Q, K, V and the n×n score matrix, scaled by √d_k.',
                     'Build the upper-triangle mask with np.triu(..., k=1) and fill those scores with -inf.',
                     'Softmax each row (your stage-2 function handles -inf), then multiply by V.',
                     'Test causality by changing a later token and comparing earlier outputs.'],
        hints=['exp(-inf) is exactly 0, so masked weights vanish with no special case.',
               'k.shape[1] is d_k; scale before masking.'],
        checks=[
            check('worked example by hand', '''
x = np.array([[1.,0.],[0.,1.],[1.,1.]]); I = np.eye(2)
out, w = causal_attention(x, I, I, I)
a = 1/(1+math.exp(1/math.sqrt(2)))
assert np.allclose(w, [[1,0,0],[a,1-a,0],[.24825,.24825,.50350]], atol=1e-4)
assert np.allclose(out[1], [a, 1-a]) and np.allclose(out[2], [.75175,.75175], atol=1e-4)
'''),
            check('masked weights are zero and rows sum to one', '''
rng = np.random.default_rng(5); x = rng.normal(size=(5,3))
wq, wk, wv = rng.normal(size=(3,2)), rng.normal(size=(3,2)), rng.normal(size=(3,4))
out, w = causal_attention(x, wq, wk, wv)
assert w.shape == (5,5) and out.shape == (5,4), 'd_model=3, d_k=2, d_v=4'
assert np.allclose(np.triu(w, 1), 0) and np.allclose(w.sum(1), 1)
q, k = x @ wq, x @ wk
for i in range(5):
    s = np.array([q[i] @ k[j] / math.sqrt(2) for j in range(i + 1)])
    e = np.exp(s - s.max())
    assert np.allclose(w[i, :i+1], e / e.sum()), 'scores are scaled by sqrt(d_k)'
'''),
            check('the future cannot change the past', '''
rng = np.random.default_rng(8); x = rng.normal(size=(6,4))
wq, wk, wv = (rng.normal(size=(4,4)) for _ in range(3))
before, _ = causal_attention(x, wq, wk, wv)
x2 = x.copy(); x2[4] += 10
after, _ = causal_attention(x2, wq, wk, wv)
assert np.allclose(before[:4], after[:4]) and not np.allclose(before[4:], after[4:])
'''),
            check('order inside the past is invisible without positions', '''
rng = np.random.default_rng(9); x = rng.normal(size=(4,3))
wq, wk, wv = (rng.normal(size=(3,2)) for _ in range(3))
out, _ = causal_attention(x, wq, wk, wv)
swapped = x[[1,0,2,3]]
out2, _ = causal_attention(swapped, wq, wk, wv)
assert np.allclose(out[2:], out2[2:]) and not np.allclose(out[0], out2[0])
'''),
        ],
        mutants=[
            ('no causal mask', 'causal_attention', '''
def causal_attention(x, wq, wk, wv):
    x = np.asarray(x, dtype=float); q, k, v = x @ wq, x @ wk, x @ wv
    w = softmax(q @ k.T / np.sqrt(k.shape[1])); return w @ v, w
'''),
            ('mask after softmax', 'causal_attention', '''
def causal_attention(x, wq, wk, wv):
    x = np.asarray(x, dtype=float); q, k, v = x @ wq, x @ wk, x @ wv
    w = softmax(q @ k.T / np.sqrt(k.shape[1])) * np.tril(np.ones((len(x), len(x)))); return w @ v, w
'''),
            ('unscaled scores', 'causal_attention', '''
def causal_attention(x, wq, wk, wv):
    x = np.asarray(x, dtype=float); q, k, v = x @ wq, x @ wk, x @ wv
    s = np.where(np.triu(np.ones((len(x), len(x)), dtype=bool), 1), -np.inf, q @ k.T)
    w = softmax(s); return w @ v, w
'''),
            ('scaled by d_model', 'causal_attention', '''
def causal_attention(x, wq, wk, wv):
    x = np.asarray(x, dtype=float); q, k, v = x @ wq, x @ wk, x @ wv
    s = np.where(np.triu(np.ones((len(x), len(x)), dtype=bool), 1), -np.inf, q @ k.T / np.sqrt(x.shape[1]))
    w = softmax(s); return w @ v, w
'''),
            ('softmax down columns', 'causal_attention', '''
def causal_attention(x, wq, wk, wv):
    x = np.asarray(x, dtype=float); q, k, v = x @ wq, x @ wk, x @ wv
    s = np.where(np.triu(np.ones((len(x), len(x)), dtype=bool), 1), -np.inf, q @ k.T / np.sqrt(k.shape[1]))
    w = softmax(s.T).T; return w @ v, w
'''),
        ]),
    dict(
        title='6 · Assemble a tiny transformer and sample',
        goal='Build a pre-norm transformer block with positions, a seeded sampler with temperature and top-k, and fit its readout.',
        prerequisites='Stages 1–5; mean and variance; random number generators with seeds.',
        lesson='''A transformer block is stage 5 plus three engineering ideas that make deep stacks trainable:
1. Positions: add a learned position vector to each token embedding, x = tok[ids] + pos[0:n]. Stage 5 showed attention can't see order otherwise.
2. Residual connections: x = x + attention(…) and then x = x + mlp(…). Each sub-layer adds a correction to a running stream instead of replacing it, so a block whose weights are zero is exactly the identity.
3. Layer normalisation (pre-norm): each sub-layer reads layer_norm(x), so every position's vector has mean 0 and variance 1 before entering. This keeps scales stable when blocks stack.
A final layer_norm, then a head matrix, gives logits for every position. Position t's logits must depend only on tokens 0…t; the causal mask makes the whole stack obey this, and a test checks it.

Generation is a loop: run the model on the last `context` ids, take the last row of logits, pick a next id and append it. Picking rules:
- temperature T divides the logits. T → 0 means always take the argmax (greedy, define T = 0 as exactly that); T = 1 is the model's distribution; T > 1 flattens it.
- top-k keeps only the k largest logits before the softmax (ties broken by lower id), so unlikely tokens can never appear.
- Use np.random.default_rng(seed) and pass it through, so the same seed in the same environment repeats exactly. A seed is not a promise across library versions.

Shapes and initialisation (the checks don't depend on your init, but the worked numbers do): with vocabulary V, context C, width d and hidden h, tok is V×d and pos is C×d, both drawn from normal(0, 0.5). wq, wk, wv and wo are d×d, w1 is d×h and head is d×V, all from normal(0, 1/√d). w2 is h×d from normal(0, 1/√h). Draw them in that order from np.random.default_rng(seed). A zero head is also a fine start, since only the head is trained here.

Honest scope for this stage: training every weight needs backpropagation through attention and layer norm, which is a stretch goal (and a natural use for PyTorch on your laptop, written by you). Here you train only the head on frozen transformer features, with your stage-3 gradient, to see the loss move. Record that clearly in your model card: “readout-only training on random features” is not a trained GPT.

Misconception: “a fluent sample proves the model is good.” Report held-out loss against the uniform (ln V) and bigram baselines, and show a sample as illustration only.

Guided activity A: set every block weight to zero and predict transformer_features. Answer: layer_norm(tok + pos), because the residuals pass it straight through. Guided activity B: generate 6 tokens at temperature 0 twice and with top_k = 1, and explain why all three agree.
On your own: the hidden checks test causality of the whole model, position sensitivity and the sampler rules.''',
        starter='''# Keep stages 1–5 above. Stage 6:

def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    raise NotImplementedError

def init_transformer(size: int, context: int, width: int = 8, hidden: int = 16, seed: int = 0) -> dict:
    # Keys and shapes: tok V×d, pos C×d (normal 0, .5); wq wk wv wo d×d, w1 d×h, head d×V (normal 0, 1/√d); w2 h×d (normal 0, 1/√h).
    raise NotImplementedError

def transformer_features(params: dict, ids: list[int]) -> np.ndarray:
    raise NotImplementedError

def transformer_logits(params: dict, ids: list[int]) -> np.ndarray:
    raise NotImplementedError

def sample_next(logits_row: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    raise NotImplementedError

def generate(params: dict, prompt: list[int], count: int, temperature: float = 1.0, top_k: int = 5, seed: int = 0) -> list[int]:
    raise NotImplementedError

def fit_head(params: dict, ids: list[int], steps: int = 200, lr: float = 0.5) -> list[float]:
    # Train only params['head'] on frozen features; return losses.
    raise NotImplementedError''',
        worked='''With init_transformer(5, 8, seed=1) and prompt [1], greedy generation (T = 0) gives [1, 0, 1, 3, 3, 3, 0] every time. With T = 1, top_k = 5 and seed 3, it gives [1, 0, 1, 0, 2, 1, 0]: the same seed always gives that sequence in this environment, and a different seed differs. Fitting only the head on [1,2,3,4]×20 (200 steps, lr 0.5) moves the training loss from ≈ 1.63 to ≈ 0.03. The random features already separate the four positions of the repeating pattern, so this shows the readout learning, not that the model understands text.''',
        walkthrough=['layer_norm over the last axis; check each row has mean ≈ 0, variance ≈ 1.',
                     'features: embeddings + positions → x + attn(LN x)·wo → x + tanh(LN x·w1)·w2 → LN.',
                     'sample_next: T = 0 → argmax; else top-k ids, softmax(logits/T), rng.choice.',
                     'generate with a sliding window of the last context ids; fit_head reuses your softmax gradient.'],
        hints=['Zero all weights except tok/pos to test the residual identity.',
               'Test causality on the whole model the way you did in stage 5.'],
        checks=[
            check('layer norm and residual identity', '''
x = np.array([[1.,2.,3.,4.],[2.,2.,2.,6.]])
y = layer_norm(x)
assert np.allclose(y.mean(1), 0) and np.allclose(y.var(1), 1, atol=1e-4)
p = init_transformer(5, 6, width=4, hidden=8, seed=0)
for k in ['wq','wk','wv','wo','w1','w2']: p[k] = np.zeros_like(p[k])
ids = [1,3,2]
assert np.allclose(transformer_features(p, ids), layer_norm(p['tok'][ids] + p['pos'][:3]))
'''),
            check('whole-model causality and position sensitivity', '''
z = init_transformer(4, 5, width=6, hidden=3, seed=9)
for k in ['wq','wk','wv','wo','w1','w2']: z[k] = np.zeros_like(z[k])
assert np.allclose(transformer_features(z, [3,0,2,2]), layer_norm(z['tok'][[3,0,2,2]] + z['pos'][:4])), 'zero sub-layers must leave the residual stream unchanged'
p = init_transformer(6, 8, seed=2)
assert transformer_logits(p, [1,2,3,4,5]).shape == (5,6)
a = transformer_features(p, [1,2,3,4,5]); b = transformer_features(p, [1,2,3,0,0])
assert np.allclose(a[:3], b[:3]) and not np.allclose(a[3:], b[3:])
c = transformer_features(p, [2,1,3])
assert not np.allclose(transformer_features(p, [1,2,3])[2], c[2]), 'positions must make order visible'
def ln(v): return (v - v.mean(-1, keepdims=True)) / np.sqrt(v.var(-1, keepdims=True) + 1e-5)
ids = [4,0,3,3,1]; x = p['tok'][ids] + p['pos'][:5]; h = ln(x); q, k, v = h @ p['wq'], h @ p['wk'], h @ p['wv']
sc = q @ k.T / np.sqrt(k.shape[1]); sc[np.triu_indices(5, 1)] = -np.inf; e = np.exp(sc - sc.max(1, keepdims=True)); att = (e / e.sum(1, keepdims=True)) @ v
x = x + att @ p['wo']; x = x + np.tanh(ln(x) @ p['w1']) @ p['w2']
assert np.allclose(transformer_features(p, ids), ln(x), atol=1e-9), 'block: x + attn(LN x)·wo, then x + tanh(LN x·w1)·w2, then LN'
try: transformer_logits(p, list(range(6))*2)
except ValueError: pass
else: raise AssertionError('reject inputs longer than the context')
'''),
            check('sampler rules: greedy, top-k, seeds', '''
row = np.array([0.5, 3.0, 2.9, -1.0])
rng = np.random.default_rng(0)
assert sample_next(row, 0, 4, rng) == 1 and all(sample_next(row, 1.0, 1, rng) == 1 for _ in range(20))
draws = {sample_next(row, 5.0, 2, np.random.default_rng(s)) for s in range(60)}
assert draws == {1, 2}, 'top-2 keeps exactly the two largest'
p = init_transformer(5, 8, seed=1)
assert generate(p, [1], 6, 0) == generate(p, [1], 6, 0) == generate(p, [1], 6, 1.0, 1)
assert generate(p, [1], 6, 1.0, 5, seed=3) == generate(p, [1], 6, 1.0, 5, seed=3)
for bad in [(-1, 2), (1, 0)]:
    try: sample_next(row, *bad, rng)
    except ValueError: pass
    else: raise AssertionError('reject negative temperature or top_k < 1')
two = np.array([0.0, 2.0])
for t in [0.5, 2.0]:
    hi = 1 / (1 + np.exp(-2.0 / t))  # independent oracle: softmax([0, 2 / T]) gives id 1 this probability
    draw = np.random.default_rng(7); share = np.mean([sample_next(two, t, 2, draw) for _ in range(20000)])
    assert abs(share - hi) < 0.03, f'T = {t}: softmax(logits / T) draws id 1 with probability {hi:.5f}, observed {share:.5f}'
'''),
            check('readout training lowers loss on a fresh pattern; long prompts slide', '''
p = init_transformer(6, 8, seed=3); seq = [2,5,3,1]*20
losses = fit_head(p, seq, 200, 0.5)
assert losses[-1] < losses[0] - 0.5
assert all(int(transformer_logits(p, seq[:k])[-1].argmax()) == seq[k] for k in range(1, 9)), 'the head must predict the NEXT token'
prompt = [5,1,4,1,2,5,1,4,1,2]; ids = prompt[:]
for _ in range(4): ids.append(int(transformer_logits(p, ids[-8:])[-1].argmax()))
assert generate(p, prompt, 4, 0) == ids, 'greedy generation reads the last row of the last 8 ids'
'''),
        ],
        mutants=[
            ('post-norm without residual', 'transformer_features', '''
def transformer_features(params, ids):
    if len(ids) == 0 or len(ids) > len(params['pos']): raise ValueError('bad')
    x = params['tok'][np.asarray(ids)] + params['pos'][:len(ids)]
    x = layer_norm(causal_attention(x, params['wq'], params['wk'], params['wv'])[0] @ params['wo'])
    x = layer_norm(np.tanh(x @ params['w1']) @ params['w2']); return x
'''),
            ('positions ignored', 'transformer_features', '''
def transformer_features(params, ids):
    if len(ids) == 0 or len(ids) > len(params['pos']): raise ValueError('bad')
    x = params['tok'][np.asarray(ids)]
    att, _ = causal_attention(layer_norm(x), params['wq'], params['wk'], params['wv']); x = x + att @ params['wo']
    x = x + np.tanh(layer_norm(x) @ params['w1']) @ params['w2']; return layer_norm(x)
'''),
            ('top-k off by one', 'sample_next', '''
def sample_next(logits_row, temperature, top_k, rng):
    if temperature < 0 or top_k < 1: raise ValueError('bad')
    row = np.asarray(logits_row, dtype=float)
    if temperature == 0: return int(row.argmax())
    keep = np.argsort(-row, kind='stable')[:top_k + 1]; probs = softmax(row[keep] / temperature)
    return int(keep[rng.choice(len(keep), p=probs)])
'''),
            ('head trained on the current token', 'fit_head', '''
def fit_head(params, ids, steps=200, lr=0.5):
    window = len(params['pos']); feats, targets = [], []
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start:start + window + 1]
        if len(chunk) < 2: break
        feats.append(transformer_features(params, chunk[:-1])); targets.extend(chunk[:-1])
    f, t = np.vstack(feats), np.asarray(targets); losses = []
    for _ in range(steps):
        logits = f @ params['head']; losses.append(cross_entropy(logits, list(t)))
        d = softmax(logits); d[np.arange(len(t)), t] -= 1; params['head'] -= lr * (f.T @ d) / len(t)
    losses.append(cross_entropy(f @ params['head'], list(t))); return losses
'''),
            ('generation never slides', 'generate', '''
def generate(params, prompt, count, temperature=1.0, top_k=5, seed=0):
    rng = np.random.default_rng(seed); ids = list(prompt); window = len(params['pos'])
    for _ in range(count): ids.append(sample_next(transformer_logits(params, ids[:window])[-1], temperature, top_k, rng))
    return ids
'''),
            ('samples from the first row', 'generate', '''
def generate(params, prompt, count, temperature=1.0, top_k=5, seed=0):
    rng = np.random.default_rng(seed); ids = list(prompt); window = len(params['pos'])
    for _ in range(count): ids.append(sample_next(transformer_logits(params, ids[-window:])[0], temperature, top_k, rng))
    return ids
'''),
            ('MLP sub-layer reads un-normalised x', 'transformer_features', '''
def transformer_features(params, ids):
    if len(ids) == 0 or len(ids) > len(params['pos']): raise ValueError('bad')
    x = params['tok'][np.asarray(ids)] + params['pos'][:len(ids)]
    att, _ = causal_attention(layer_norm(x), params['wq'], params['wk'], params['wv']); x = x + att @ params['wo']
    x = x + np.tanh(x @ params['w1']) @ params['w2']; return layer_norm(x)
'''),
            ('negative temperature accepted', 'sample_next', '''
def sample_next(logits_row, temperature, top_k, rng):
    if top_k < 1: raise ValueError('bad')
    row = np.asarray(logits_row, dtype=float); keep = np.argsort(-row, kind='stable')[:top_k]
    probs = softmax(row[keep] / max(temperature, 1e-9) if temperature >= 0 else row[keep])
    return int(keep[rng.choice(len(keep), p=probs)])
'''),
            ('temperature multiplies the logits', 'sample_next', '''
def sample_next(logits_row, temperature, top_k, rng):
    if temperature < 0 or top_k < 1: raise ValueError('bad')
    row = np.asarray(logits_row, dtype=float)
    if temperature == 0: return int(row.argmax())
    keep = np.argsort(-row, kind='stable')[:top_k]; probs = softmax(row[keep] * temperature)
    return int(keep[rng.choice(len(keep), p=probs)])
'''),
            ('temperature multiplies, inverse-CDF draw', 'sample_next', '''
def sample_next(logits_row, temperature, top_k, rng):
    if temperature < 0 or top_k < 1: raise ValueError('bad')
    row = np.asarray(logits_row, dtype=float)
    if temperature == 0: return int(row.argmax())
    keep = np.argsort(-row, kind='stable')[:top_k]; cdf = np.cumsum(softmax(row[keep] * temperature))
    return int(keep[min(int(np.searchsorted(cdf, rng.random(), side='right')), len(keep) - 1)])
'''),
        ]),
]

README_TEMPLATE = '''# <your project name>: a GPT built from scratch, stage by stage

> One line: what it is and what you measured. Example shape: “A NumPy GPT-style model I wrote from tokenizer to transformer; held-out loss <your number> vs <your baseline>.” Use only numbers from your own runs.

![demo](docs/demo.gif) <!-- record your own: a terminal session of generate() or a small web demo -->

## Quickstart
```sh
python -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
pytest
python -m <your_module> --text <your-permitted-text.txt> --seed 0
```

## What I built (one stage = a few focused commits)
| Stage | Component | My tests prove | Commit(s) |
| --- | --- | --- | --- |
| 1 | BPE tokenizer | lossless round trip, tie-break, overlap | |
| 2 | Softmax + cross-entropy | stability at large logits, ln V baseline | |
| 3 | Gradient-descent bigram | finite-difference gradient check | |
| 4 | Embeddings + MLP | gradient check incl. repeated tokens; context beats bigram on dev | |
| 5 | Causal self-attention | future can't change the past | |
| 6 | Transformer block + sampler | whole-model causality, seeded sampling | |

## Architecture
```mermaid
flowchart LR
  T[text] --> B[BPE encode] --> E[token + position embeddings]
  E --> A["x + attention(LN x)"] --> M["x + MLP(LN x)"] --> N[LN] --> H[head → logits]
  H --> S["sampler (temperature, top-k, seed)"]
```

## Results (measured: fill in from your own runs)
| Model | Train loss | Dev loss | Test loss | Where it ran (device, versions) |
| --- | --- | --- | --- | --- |
| Uniform (ln V) | | | | |
| Bigram (stage 3) | | | | |
| MLP (stage 4) | | | | |
| Transformer (readout-only or full PyTorch: say which) | | | | |

Sample (seed, temperature, top-k stated). An illustration, not evidence of quality.

## Design decisions and trade-offs
- Tokenizer: why BPE, merge count, spaces as symbols (no pre-splitting) and what that costs.
- Why pre-norm residual blocks; context length; width.
- What I trained (readout only vs every weight) and why.

## Limits
- Data size and permission; what the model can't do; bias and privacy notes (see MODEL-CARD.md).

## What I learned
- One bug that a test caught, the counterexample, and the fix.

## Next steps
- See ROADMAP.md (e.g. KV cache, full backprop, a loss-curve study on the laptop GPU).

License: <your choice, e.g. MIT>.
'''

FOLLOW_ON = '''# Try without a supplied solution: attention that sorts

Build a different small model with the same parts: given a sequence of 6 digits followed by a separator, output the digits in sorted order, one token at a time. Generate your own synthetic train/dev/test splits (make sure no test sequence appears in training). Decide your tokenizer (digits plus separator, no BPE), context length and whether you need positions. Before training, write tests: causality of your model; a check that your data splits don't overlap; and a baseline (e.g. always predicting the most common digit). Report the exact-sequence accuracy and the per-token loss on test against that baseline, with where it ran. Explain one failure case (duplicates? the first output position?). No reference implementation, worked solution or promised accuracy is supplied.
'''


def taught_gpt_series():
    stages = []
    for number, spec in enumerate(STAGES, start=1):
        checks = spec['checks']
        stages.append(dict(
            id=f'gpt2-{number}', title=spec['title'], goal=spec['goal'],
            lessonRef=f'projects/gpt2/{number}',
            lesson='Prerequisites: ' + spec['prerequisites'] + '\n\n' + spec['lesson'],
            language='python',
            starterFiles={'src/main.py': HEADER + dedent(spec['starter']).strip() + '\n'},
            referenceFiles={'src/main.py': HEADER + CUMULATIVE[number - 1].strip() + '\n'},
            visibleTests=checks[:1], hiddenTests=checks[1:], hints=spec['hints'],
            reviewRubric=RUBRIC, workedExample=spec['worked'], walkthrough=spec['walkthrough'],
            concepts=[spec['goal'], 'Learn freely, predict by hand, build with guidance, then check your own version.'],
            guidedPrompt='Change one input from the worked example. Predict the result by hand, then run your own code and explain any difference. Hints are free.',
            requiresHostProcess=False, prerequisites=spec['prerequisites']))
    return dict(
        id='gpt2', title='Build your own GPT, taught (v2)',
        outline=['BPE tokenizer', 'Softmax and cross-entropy', 'Gradient-descent bigram', 'Embeddings + MLP context', 'Causal self-attention', 'Tiny transformer and sampler'],
        complete=True, stages=stages,
        resumeBullets=['Only after your own run: built a GPT-style model from scratch in NumPy (BPE → attention → transformer) with gradient-checked layers; held-out loss [your number] vs uniform [ln V] and bigram [your number].',
                       'Caught [your bug] with [your test, e.g. a finite-difference check or causality test] and documented the fix.'],
        demoRecipe='Python in the app remains HOLD. Run your own code on your Mac or laptop: record `generate()` with a stated seed/temperature/top-k in a terminal recording, or a small local web demo. Put measured losses (yours) in the README. Authored lesson numbers are not your results.',
        stretchGoals=['Full backpropagation through attention and layer norm in NumPy, gradient-checked.', 'A KV cache for generation, with a test that cached and uncached logits match.', 'Write your own PyTorch version on the laptop (CPU or GPU) that trains every weight; check it against your NumPy forward pass, then do a loss-vs-steps and loss-vs-width study with your own plots. No PyTorch file is supplied.', 'Unsolved follow-on: FOLLOW-ON.md (attention that sorts).'],
        seniorReviewRubric=['Gradient checks exist for every hand-written backward pass and use a tolerance that would catch a missing /N.', 'Causality is tested on the whole model, not only on attention.', 'Train/dev/test separation is explicit and the baselines (ln V, bigram) are reported honestly, even when they win.', 'The README separates readout-only from full training and states where each run happened.', 'Commit history shows a few focused commits per stage.'],
        prerequisites=['Python functions, lists, dicts', 'NumPy shapes, broadcasting and axis arguments', 'Derivatives of exp, ln and tanh; the chain rule', 'Recommended first: tinylm2 (tokenizer, bigram, held-out metrics)'],
        authoringStatus='Original source fixtures, references and checks. Every reference passes its checks and every named wrong implementation is rejected by a hidden check (host test). No app execution, learner outcome, mastery or runtime admission is implied.',
        predecessor='gpt (legacy identities preserved, no progress migration)',
        supplementalFiles={'README-TEMPLATE.md': README_TEMPLATE, 'FOLLOW-ON.md': FOLLOW_ON,
                           'MODEL-CARD.md': '# Model card\n\nData origin and permission:\nTokenizer (merges, vocabulary size):\nTrain / dev / test split and counts:\nArchitecture (context, width, parameters) and what was trained (readout only or every weight):\nWhere it ran (device, Python/NumPy/PyTorch versions), seed, steps:\nTrain / dev / test loss vs uniform ln V and bigram:\nSample (seed, temperature, top-k), labelled as illustration:\nFailure modes, bias and privacy notes, intended use:\n'})


MUTANTS = {f'gpt2-{n}': spec['mutants'] for n, spec in enumerate(STAGES, start=1)}
