"""Reference implementations for the taught GPT series (gpt2). Host authoring data only.

Each stage's reference file is cumulative: the stage-N reference contains every function the
learner has written up to stage N, because checks run against the learner's whole src/main.py.
"""

BPE = '''
def pair_counts(symbols: list[str]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for left, right in zip(symbols, symbols[1:]):
        counts[(left, right)] = counts.get((left, right), 0) + 1
    return counts

def merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    merged, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1]); i += 2
        else:
            merged.append(symbols[i]); i += 1
    return merged

def train_bpe(text: str, merges: int) -> list[tuple[str, str]]:
    if not isinstance(merges, int) or merges < 0:
        raise ValueError('Ask for a non-negative whole number of merges')
    symbols, learned = list(text), []
    for _ in range(merges):
        counts = pair_counts(symbols)
        best = max(counts.values(), default=0)
        if best < 2:
            break
        pair = min(p for p, c in counts.items() if c == best)
        learned.append(pair)
        symbols = merge_pair(symbols, pair)
    return learned

def encode_bpe(text: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols = list(text)
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return symbols

def decode_bpe(tokens: list[str]) -> str:
    return ''.join(tokens)
'''

SOFTMAX = '''
def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(logits: np.ndarray, targets: list[int]) -> float:
    z = np.asarray(logits, dtype=float)
    t = np.asarray(targets)
    if z.ndim != 2 or t.shape != (z.shape[0],) or len(t) == 0:
        raise ValueError('Use one row of logits per target')
    if t.dtype.kind not in 'iu' or (t < 0).any() or (t >= z.shape[1]).any():
        raise ValueError('Targets must be class indices')
    shifted = z - z.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(t)), t].mean())
'''

TRAIN = '''
def bigram_loss_and_grad(w: np.ndarray, ids: list[int]) -> tuple[float, np.ndarray]:
    x, y = np.asarray(ids[:-1]), np.asarray(ids[1:])
    if len(y) == 0:
        raise ValueError('Need at least one transition')
    logits = w[x]
    p = softmax(logits)
    loss = cross_entropy(logits, y.tolist())
    d_logits = p.copy()
    d_logits[np.arange(len(y)), y] -= 1
    d_logits /= len(y)
    grad = np.zeros_like(w, dtype=float)
    np.add.at(grad, x, d_logits)
    return loss, grad

def train_bigram(ids: list[int], size: int, steps: int = 300, lr: float = 1.0) -> tuple[np.ndarray, list[float]]:
    w = np.zeros((size, size))
    losses = []
    for _ in range(steps):
        loss, grad = bigram_loss_and_grad(w, ids)
        losses.append(loss)
        w -= lr * grad
    losses.append(bigram_loss_and_grad(w, ids)[0])
    return w, losses
'''

MLP = '''
def make_contexts(ids: list[int], block: int) -> tuple[np.ndarray, np.ndarray]:
    if block < 1:
        raise ValueError('Context needs at least one token')
    padded = [0] * block + list(ids)
    contexts = [padded[i:i + block] for i in range(len(ids))]
    return np.array(contexts, dtype=int).reshape(len(ids), block), np.array(ids, dtype=int)

def init_mlp(size: int, block: int, embed: int = 4, hidden: int = 16, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {'E': rng.normal(0, 0.5, (size, embed)),
            'W1': rng.normal(0, 1 / np.sqrt(block * embed), (block * embed, hidden)), 'b1': np.zeros(hidden),
            'W2': rng.normal(0, 1 / np.sqrt(hidden), (hidden, size)), 'b2': np.zeros(size)}

def mlp_forward(params: dict, contexts: np.ndarray) -> tuple[np.ndarray, dict]:
    emb = params['E'][contexts]
    flat = emb.reshape(len(contexts), -1)
    h = np.tanh(flat @ params['W1'] + params['b1'])
    logits = h @ params['W2'] + params['b2']
    return logits, {'contexts': contexts, 'flat': flat, 'h': h}

def mlp_loss_and_grads(params: dict, contexts: np.ndarray, targets: np.ndarray) -> tuple[float, dict]:
    logits, cache = mlp_forward(params, contexts)
    loss = cross_entropy(logits, list(targets))
    d = softmax(logits)
    d[np.arange(len(targets)), targets] -= 1
    d /= len(targets)
    grads = {'W2': cache['h'].T @ d, 'b2': d.sum(0)}
    dh = (d @ params['W2'].T) * (1 - cache['h'] ** 2)
    grads['W1'] = cache['flat'].T @ dh
    grads['b1'] = dh.sum(0)
    d_emb = (dh @ params['W1'].T).reshape(*contexts.shape, -1)
    grads['E'] = np.zeros_like(params['E'])
    np.add.at(grads['E'], contexts, d_emb)
    return loss, grads

def train_mlp(params: dict, contexts: np.ndarray, targets: np.ndarray, steps: int = 400, lr: float = 0.5) -> list[float]:
    losses = []
    for _ in range(steps):
        loss, grads = mlp_loss_and_grads(params, contexts, targets)
        losses.append(loss)
        for name in params:
            params[name] -= lr * grads[name]
    return losses
'''

ATTENTION = '''
def causal_attention(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    q, k, v = x @ wq, x @ wk, x @ wv
    scores = q @ k.T / np.sqrt(k.shape[1])
    n = len(x)
    future = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)
    weights = softmax(scores)
    return weights @ v, weights
'''

TRANSFORMER = '''
def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def init_transformer(size: int, context: int, width: int = 8, hidden: int = 16, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    s = 1 / np.sqrt(width)
    return {'tok': rng.normal(0, 0.5, (size, width)), 'pos': rng.normal(0, 0.5, (context, width)),
            'wq': rng.normal(0, s, (width, width)), 'wk': rng.normal(0, s, (width, width)), 'wv': rng.normal(0, s, (width, width)),
            'wo': rng.normal(0, s, (width, width)),
            'w1': rng.normal(0, s, (width, hidden)), 'w2': rng.normal(0, 1 / np.sqrt(hidden), (hidden, width)),
            'head': rng.normal(0, s, (width, size))}

def transformer_features(params: dict, ids: list[int]) -> np.ndarray:
    if len(ids) == 0 or len(ids) > len(params['pos']):
        raise ValueError('Use between one token and the context length')
    x = params['tok'][np.asarray(ids)] + params['pos'][:len(ids)]
    attended, _ = causal_attention(layer_norm(x), params['wq'], params['wk'], params['wv'])
    x = x + attended @ params['wo']
    x = x + np.tanh(layer_norm(x) @ params['w1']) @ params['w2']
    return layer_norm(x)

def transformer_logits(params: dict, ids: list[int]) -> np.ndarray:
    return transformer_features(params, ids) @ params['head']

def sample_next(logits_row: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    if temperature < 0 or top_k < 1:
        raise ValueError('Use temperature >= 0 and top_k >= 1')
    row = np.asarray(logits_row, dtype=float)
    if temperature == 0:
        return int(row.argmax())
    keep = np.argsort(-row, kind='stable')[:top_k]
    probs = softmax(row[keep] / temperature)
    return int(keep[rng.choice(len(keep), p=probs)])

def generate(params: dict, prompt: list[int], count: int, temperature: float = 1.0, top_k: int = 5, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    ids = list(prompt)
    window = len(params['pos'])
    for _ in range(count):
        ids.append(sample_next(transformer_logits(params, ids[-window:])[-1], temperature, top_k, rng))
    return ids

def fit_head(params: dict, ids: list[int], steps: int = 200, lr: float = 0.5) -> list[float]:
    window = len(params['pos'])
    feats, targets = [], []
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start:start + window + 1]
        if len(chunk) < 2:
            break
        feats.append(transformer_features(params, chunk[:-1]))
        targets.extend(chunk[1:])
    f, t = np.vstack(feats), np.asarray(targets)
    losses = []
    for _ in range(steps):
        logits = f @ params['head']
        losses.append(cross_entropy(logits, list(t)))
        d = softmax(logits)
        d[np.arange(len(t)), t] -= 1
        params['head'] -= lr * (f.T @ d) / len(t)
    losses.append(cross_entropy(f @ params['head'], list(t)))
    return losses
'''

CUMULATIVE = [BPE, BPE + SOFTMAX, BPE + SOFTMAX + TRAIN, BPE + SOFTMAX + TRAIN + MLP,
              BPE + SOFTMAX + TRAIN + MLP + ATTENTION, BPE + SOFTMAX + TRAIN + MLP + ATTENTION + TRANSFORMER]
