"""Original v2 project fixtures. Does not execute learner code or admit a runtime."""
from textwrap import dedent

BIGRAM_REFERENCE = '''
def bigram(ids: list[int], size: int, k: float = 1.0) -> np.ndarray:
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError('Use a positive vocabulary size')
    if not math.isfinite(k) or k < 0:
        raise ValueError('Smoothing must be finite and nonnegative')
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= size for i in ids):
        raise ValueError('Token outside the training vocabulary')
    counts = np.full((size, size), k, dtype=float)
    for position in range(len(ids) - 1):
        counts[ids[position], ids[position + 1]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.full_like(counts, 1 / size), where=totals > 0)
'''
# Hand-counted rational expectations, independent of the NumPy reference.
BIGRAM_CASES = [
    ('overlap and row totals', [0, 0, 0, 1, 1, 0], 2, [[(3, 5), (2, 5)], [(1, 2), (1, 2)]]),
    ('directional identical rows', [0, 1, 1], 2, [[(1, 3), (2, 3)], [(1, 3), (2, 3)]]),
    ('three symbols and unseen row', [0, 1, 1, 0], 3, [[(1, 4), (1, 2), (1, 4)], [(2, 5), (2, 5), (1, 5)], [(1, 3), (1, 3), (1, 3)]]),
]

def bigram_checks():
    checks = []
    for name, ids, size, rows in BIGRAM_CASES:
        expected = [[f'{a}/{b}' for a, b in row] for row in rows]
        literal = '[' + ','.join('[' + ','.join(row) + ']' for row in expected) + ']'
        checks.append(dict(name=name, source=f'p=bigram({ids!r}, {size}); assert p.shape==({size},{size}); assert np.allclose(p, np.array({literal}), atol=1e-12, rtol=0); assert np.allclose(p.sum(axis=1),1,atol=1e-12,rtol=0)'))
    checks.append(dict(name='empty and corrected audit examples', source='assert np.allclose(bigram([],3),1/3); assert np.allclose(bigram([0,0,1],2),.5); assert np.allclose(bigram([0,1,1],2),[[1/3,2/3],[1/3,2/3]])'))
    # Extend the existing policy check: same milestone and receipt check count.
    checks[-1]['source'] += "\np=bigram([0,0,0,1,1,0],2,.5); assert np.isfinite(p).all(); assert np.allclose(p,[[5/8,3/8],[1/2,1/2]],atol=1e-12,rtol=0)\np=bigram([0,0,1],3,0); assert np.isfinite(p).all(); assert np.allclose(p,[[1/2,1/2,0],[1/3,1/3,1/3],[1/3,1/3,1/3]],atol=1e-12,rtol=0); assert p[0,2]==0"
    return checks


def make_stage(number, title, goal, lesson, starter, reference, checks, worked, walkthrough, prerequisites, hints):
    return dict(id=f'tinylm2-{number}', title=title, goal=goal,
        lessonRef=f'projects/tinylm2/{number}', lesson='Prerequisites: '+prerequisites+'\n\n'+lesson,
        language='python', starterFiles={'src/main.py':dedent(starter).strip()+'\n'},
        referenceFiles={'src/main.py':'import math\nimport numpy as np\n'+dedent(reference).strip()+'\n'},
        visibleTests=checks[:1], hiddenTests=checks[1:], hints=hints,
        reviewRubric=['Explain your prediction before showing a result.','Show an independently computed boundary case.','Explain a revision and its counterexample.','Describe help used neutrally; source authored is not source executed.'],
        workedExample=worked, walkthrough=walkthrough,
        concepts=[goal, 'Learn freely, predict, build with guidance, then check your own version.'],
        guidedPrompt='Use a different input from the worked example. Predict and hand-trace it before running your own code.',
        requiresHostProcess=False, prerequisites=prerequisites)


def bigram_stage(reference=BIGRAM_REFERENCE):
    return make_stage(2, '2 · Count directed, overlapping transitions',
        'Build and explain a row-normalized add-k transition matrix.',
        'Rows mean current token; columns mean next token. Every adjacent pair counts, including overlapping pairs. Add k to each cell, then divide each row by its own total: P[j|i]=(C[i,j]+k)/(sum_j C[i,j]+kV). For k>0 every in-vocabulary next token has positive probability. This does not imply a lower held-out loss than uniform. With k=0 an entirely unseen row uses an explicit uniform fallback; an unseen transition in a seen row can still have probability zero. Unknown characters are handled by the tokenizer policy, not by growing V here.\n\nCounterexample: [0,0,1] produces a uniform matrix, so transposing its probabilities reveals nothing. For [0,1,1], both rows are [1/3,2/3], not [1/4,3/4] in the second row. We need several asymmetric fixtures and unequal row totals to diagnose orientation and broadcasting.\n\nGuided activity A: draw counts for [0,0,0,1,1,0] and mark five adjacent arrows. Guided activity B: add a third, unseen symbol and compare its prior-only row with rows that have observations. Keep exact fractions until the final decimal display.',
        'import numpy as np\n\n# Keep your previous learner-authored functions above this new function.\ndef bigram(ids: list[int], size: int, k: float = 1.0) -> np.ndarray:\n    raise NotImplementedError', reference, bigram_checks(),
        'For [0,0,0,1,1,0], observed counts are [[2,1],[1,1]]. Add one: [[3,2],[2,2]]. Row totals are 5 and 4, giving [[3/5,2/5],[1/2,1/2]]. A column-normalized table instead has denominators 5 and 4 across columns and does not describe next-token distributions per current token.',
        ['Draw one arrow for each adjacent pair, not one arrow per disjoint pair.','Add the same k to every cell, including unseen transitions.','Write the denominator with a row label and retain its column dimension.','Check exact asymmetric entries as well as row sums; row sums alone cannot prove the right model.'],
        'Python indexing/loops and ValueError; stage1 training-only vocabulary; 2D NumPy shapes, conditional probability and fractions.',
        ['List all len(ids)-1 directed transitions before counting.','A (V,1) row-total array divides every column of a row by the same denominator.'])


TOKENIZER_REFERENCE = '''
def split_text(text: str, train_count: int, dev_count: int) -> tuple[str, str, str]:
    if train_count < 2 or dev_count < 2 or len(text) - train_count - dev_count < 2:
        raise ValueError('Each split needs at least two characters for a transition')
    return text[:train_count], text[train_count:train_count+dev_count], text[train_count+dev_count:]

def build_vocab(training: str) -> list[str]:
    if not training:
        raise ValueError('Training text must not be empty')
    return ['<UNK>'] + sorted(set(training))

def encode(text: str, vocab: list[str], unknown: str = 'replace') -> list[int]:
    if not vocab or vocab[0] != '<UNK>' or len(set(vocab)) != len(vocab) or unknown not in ['replace', 'error']:
        raise ValueError('Use a frozen vocabulary with <UNK> at zero and a declared unknown policy')
    lookup = {character: i for i, character in enumerate(vocab)}
    result = []
    for character in text:
        if character not in lookup and unknown == 'error':
            raise ValueError('Character outside training vocabulary')
        result.append(lookup.get(character, 0))
    return result

def decode(ids: list[int], vocab: list[str]) -> str:
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= len(vocab) for i in ids):
        raise ValueError('Unknown token ID')
    return ''.join(vocab[i] for i in ids)
'''
METRICS_REFERENCE = '''
def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[0] != p.shape[1] or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError('Expected a finite square table of row distributions')
    return p

def metrics(ids: list[int], probabilities: np.ndarray) -> dict:
    p = validate_probabilities(probabilities)
    if len(ids) < 2 or any(i < 0 or i >= len(p) for i in ids):
        raise ValueError('Need at least one in-vocabulary transition')
    observed = [p[a, b] for a, b in zip(ids, ids[1:])]
    loss = math.inf if any(value == 0 for value in observed) else -sum(math.log(value) for value in observed) / len(observed)
    return {'cross_entropy': float(loss), 'perplexity': math.exp(loss), 'transitions': len(observed)}

def unigram(ids: list[int], size: int, k: float = 1.0) -> np.ndarray:
    if size < 1 or k <= 0 or not math.isfinite(k) or any(i < 0 or i >= size for i in ids):
        raise ValueError('Use in-vocabulary IDs and positive smoothing')
    frequencies = np.full(size, k, dtype=float)
    for token in ids:
        frequencies[token] += 1
    distribution = frequencies / frequencies.sum()
    return np.tile(distribution, (size, 1))
'''
EXPERIMENT_REPORT_CONTRACT = """Return a dict with these exact keys and shapes (metric means the three-key dict below):
metric = {"cross_entropy": float in nats, "perplexity": float, "transitions": int}
{
  "vocabulary": list[str], with "<UNK>" first and sorted unique training characters after it,
  "vocabulary_fit_split": "train",
  "unknown_policy": "replace unseen characters with <UNK>",
  "characters": {"train": int, "dev": int, "test": int},
  "unknown_characters": {"train": int, "dev": int, "test": int},
  "chosen_k": float,
  "selection_split": "dev",
  "dev_candidates": [{"k": float, "cross_entropy": float}, ...],
  "metrics": {"train": metric, "dev": metric, "test": metric},
  "baselines": {"uniform": metric on test, "unigram": metric on test},
  "context_tokens": 1,
  "execution_location": the caller's actual execution_location string
}
Deduplicate candidates and sort them ascending by numeric k before scoring/reporting.
The dev_candidates list follows that ascending order regardless of caller order.
For equal dev loss choose the smaller k. Unigram uses its declared default add-one smoothing.
This is the output interface; source examples here are authored, not measured results.
"""

EXPERIMENT_REFERENCE = '''
def experiment(training: str, development: str, test: str, candidates: list[float], execution_location: str) -> dict:
    if any(len(text) < 2 for text in [training, development, test]) or not execution_location.strip():
        raise ValueError('Declare splits and actual execution location')
    choices = sorted(set(candidates))
    if not choices or any(not math.isfinite(k) or not 0 < k <= 1 for k in choices):
        raise ValueError('Choose positive smoothing at most one, before inspecting the test result')
    vocab = build_vocab(training)
    encoded = {'train': encode(training, vocab, 'error'), 'dev': encode(development, vocab), 'test': encode(test, vocab)}
    size = len(vocab)
    dev_candidates = [{'k': k, 'cross_entropy': metrics(encoded['dev'], bigram(encoded['train'], size, k))['cross_entropy']} for k in choices]
    chosen = min(dev_candidates, key=lambda item: (item['cross_entropy'], item['k']))['k']
    probabilities = bigram(encoded['train'], size, chosen)
    uniform = np.full((size, size), 1 / size)
    frequency_baseline = unigram(encoded['train'], size)
    return {
        'vocabulary': vocab, 'vocabulary_fit_split': 'train', 'unknown_policy': 'replace unseen characters with <UNK>',
        'characters': {name: len(ids) for name, ids in encoded.items()},
        'unknown_characters': {name: ids.count(0) for name, ids in encoded.items()},
        'chosen_k': chosen, 'selection_split': 'dev', 'dev_candidates': dev_candidates,
        'metrics': {name: metrics(ids, probabilities) for name, ids in encoded.items()},
        'baselines': {'uniform': metrics(encoded['test'], uniform), 'unigram': metrics(encoded['test'], frequency_baseline)},
        'context_tokens': 1, 'execution_location': execution_location,
    }
'''
SAMPLING_REFERENCE = '''
def sample_ids(probabilities: np.ndarray, start: int, count: int, seed: int) -> list[int]:
    p = validate_probabilities(probabilities)
    if start < 0 or start >= len(p) or count < 0:
        raise ValueError('Use a valid seed token and nonnegative sample length')
    random = np.random.default_rng(seed)
    result = [start]
    for _ in range(count):
        result.append(int(random.choice(len(p), p=p[result[-1]])))
    return result
'''
NORMALIZE_REFERENCE = '''
def normalize_counts(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=float)
    if c.ndim != 2 or c.shape[0] == 0 or c.shape[0] != c.shape[1] or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError('Expected finite nonnegative square counts')
    totals = c.sum(axis=1, keepdims=True)
    return np.divide(c, totals, out=np.full_like(c, 1 / len(c)), where=totals > 0)
'''
EXPORT_REFERENCE = '''
def export_artifact(directory: Path, report: dict, card: dict, readme: str) -> list[str]:
    fields = ['data_origin', 'data_permission', 'split_policy', 'unknown_policy', 'context_limit', 'failure_case', 'revision_note', 'help_used', 'execution_location', 'reproduction_command']
    if any(not isinstance(card.get(name), str) or not card[name].strip() or 'TODO' in card[name] for name in fields):
        raise ValueError('Write the model card in your own words')
    if card.get('metrics') != report['metrics'] or card.get('chosen_k') != report['chosen_k'] or card['execution_location'] != report['execution_location']:
        raise ValueError('Card must report the actual experiment without changing its measurements')
    headings = ['## Demo', '## Quickstart', '## Results', '## Decisions', '## Limits', '## Revision', '## Follow-on']
    if 'TODO' in readme or any(heading not in readme for heading in headings):
        raise ValueError('Complete the README using your own explanation and measured results')
    files = {'results.json': json.dumps(report, indent=2, allow_nan=False), 'model-card.json': json.dumps(card, indent=2, allow_nan=False), 'README.md': readme}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in files):
        raise ValueError('Keep existing artifacts; choose a new export directory')
    for name, text in files.items():
        (directory / name).write_text(text, encoding='utf-8')
    return list(files)
'''
LEARNER_TEST_REFERENCE = '''
# The CLI stage harness executes source and checks in one namespace.
# pytest imports the learner module normally through the scaffold's src path.
if __name__ != '__main__':
    import math
    import numpy as np
    from main import build_vocab, encode, bigram, metrics

def test_training_vocabulary_is_frozen():
    vocab = build_vocab('ba ')
    assert vocab == ['<UNK>', ' ', 'a', 'b']
    assert encode('z', vocab) == [0]
    assert 'z' not in vocab

def test_asymmetric_transition_values():
    actual = bigram([0, 0, 0, 1, 1, 0], 2)
    assert np.allclose(actual, [[3/5, 2/5], [1/2, 1/2]], rtol=0, atol=1e-12)

def test_unseen_transition_has_infinite_unsmoothed_loss():
    p = bigram([0, 0], 2, 0)
    assert math.isinf(metrics([0, 1], p)['cross_entropy'])
'''


def check(name, source):
    return dict(name=name, source=dedent(source).strip())


def _full_series():
    first = TOKENIZER_REFERENCE
    second = first + BIGRAM_REFERENCE
    third = second + METRICS_REFERENCE
    fourth = third + EXPERIMENT_REFERENCE
    fifth = fourth + SAMPLING_REFERENCE
    sixth = fifth.replace('return np.divide(counts, totals, out=np.full_like(counts, 1 / size), where=totals > 0)', 'return normalize_counts(counts)').replace('    totals = counts.sum(axis=1, keepdims=True)\n', '') + NORMALIZE_REFERENCE
    source_imports = 'import json\nfrom pathlib import Path\n'
    stages = []
    def add(number, title, goal, lesson, starter, reference, tests, worked, walkthrough, prerequisites, hints):
        # Each starter asks the learner to retain/copy their OWN earlier functions;
        # it never installs an earlier reference solution in the active project.
        stages.append(make_stage(number, title, goal, lesson, starter, source_imports+reference, tests, worked, walkthrough, prerequisites, hints))
        return stages[-1]
    add(1, '1 · Freeze a training-only alphabet', 'Split text before fitting a tokenizer and state an unknown-character policy.',
        'Python on iPad remains on HOLD. These are authored lessons and optional reference sources, not a claim that your code has run. Use an explicitly chosen local host CLI for real checks. No notes, credentials or downloaded corpus are needed: start with synthetic text you can inspect.\n\nA model is learned from training text. Development text helps choose settings; test text measures the frozen choice. Split before learning an alphabet. If you build the vocabulary from the whole text, you have let future evaluation data change your model. A character tokenizer is a list of symbols plus a dictionary mapping each symbol to an integer. Sorting makes IDs reproducible; spaces and newlines are symbols, not cleanup. Reserve <UNK> at index0. In replace mode, map a new character to0 without changing the vocabulary. In error mode, reject it clearly. Decoding an unknown produces the visible marker <UNK>, so only known text promises an exact round trip.\n\nGuided activity A: split "ab ab\\nzz" into train4/dev2/test2 and list exactly which symbols training contains. Guided activity B: train on "λ λ\\n", predict how "λ🚁" encodes in replace and error modes. Explain why letting the helicopter add a new vocabulary slot leaks evaluation information. Independent milestone: handle a new Unicode/whitespace example from the checks without adding symbols. Skills are taught here; there is no assumed prior AI expertise.',
        '''# Implement these yourself; the worked example explains a different input.
def split_text(text: str, train_count: int, dev_count: int) -> tuple[str, str, str]:
    raise NotImplementedError

def build_vocab(training: str) -> list[str]:
    raise NotImplementedError

def encode(text: str, vocab: list[str], unknown: str = 'replace') -> list[int]:
    raise NotImplementedError

def decode(ids: list[int], vocab: list[str]) -> str:
    raise NotImplementedError''', first,
        [check('training-only split and explicit unknown', "train,dev,test=split_text('ab ab\\nzz',4,2); assert (train,dev,test)==('ab a','b\\n','zz'); v=build_vocab(train); before=list(v); assert v==['<UNK>',' ','a','b']; assert encode(test,v)==[0,0]; assert v==before"),
         check('Unicode, newline and rejection', "v=build_vocab('λ λ\\n'); assert decode(encode('λ λ\\n',v),v)=='λ λ\\n'; assert decode([0],v)=='<UNK>'\ntry: encode('🚁',v,'error')\nexcept ValueError: pass\nelse: raise AssertionError('unknown must be explicit')\nfor ids in [[-1],[len(v)]]:\n    try: decode(ids,v)\n    except ValueError: pass\n    else: raise AssertionError('invalid ID')"),
         check('independent Unicode split and frozen vocabulary', "train,dev,test=split_text('猫\\t猫 Ω\\nββ🙂猫',5,3); assert (train,dev,test)==('猫\\t猫 Ω','\\nββ','🙂猫'); v=build_vocab(train); before=list(v); assert v==['<UNK>','\\t',' ','Ω','猫']; assert encode(train,v)==[4,1,4,2,3]; assert encode(dev,v)==[0,0,0]; assert encode(test,v)==[0,4]; assert decode(encode(test,v),v)=='<UNK>猫'; assert v==before\ntry: encode(test,v,'error')\nexcept ValueError: pass\nelse: raise AssertionError('reject unseen evaluation symbol')\nassert v==before"),
         check('split boundaries and invalid policy', "try: split_text('abcd',2,1)\nexcept ValueError: pass\nelse: raise AssertionError('each split needs a pair')\ntry: encode('a',build_vocab('a'),'invent')\nexcept ValueError: pass\nelse: raise AssertionError('declare a supported policy')")],
        'Train on "aba ": vocabulary is [<UNK>, space, a, b], so "ba " encodes [3,2,1]. The new character "z" maps to0 in replace mode and raises an error in error mode. The vocabulary stays the same in both cases.',
        ['Draw train/dev/test boundaries before creating any dictionary.','Sort unique training characters, prepend the reserved unknown marker, and write the index table.','Trace encoding with known and unknown symbols; state which round trip is lossless.','Explain the split and policy to someone else before implementing.'],
        'Python functions, lists, dictionaries, slicing, loops, exceptions. No gradients or neural-network knowledge required.',
        ['The tokenizer accepts training text only, not all three splits.','Do not strip whitespace or extend the dictionary during encoding.'])
    two = bigram_stage(second)
    two['referenceFiles']['src/main.py'] = 'import math\nimport numpy as np\n'+source_imports+second
    stages.append(two)
    add(3, '3 · Measure surprise on held-out transitions', 'Implement cross-entropy, perplexity and two honest baselines.',
        'A next-token probability is judged on an observed transition. Surprise in natural-log units (nats) is -ln(p). Cross-entropy is the mean over N=len(ids)-1 transitions, not the number of characters. Perplexity exp(loss) translates that mean back into an effective number of choices. A uniform V-token baseline has loss ln(V) and perplexity V. A unigram baseline counts training token frequencies, adds k and repeats that same next-token distribution for every current token; it ignores order.\n\nDo not score the same pairs used to fit and call that held-out performance. A zero-probability observed transition has infinite loss; silently adding epsilon changes the reported model. For positive k, in-vocabulary transitions have nonzero probability, but smoothing does not guarantee a win over uniform. An unobserved context at k=0 uses the explicitly stated uniform fallback; a missing transition from an observed context can still give infinite loss. Unknown characters must already be mapped by stage1.\n\nGuided activity A: compute a two-transition loss from probabilities1/2 and1/4: -(ln(1/2)+ln(1/4))/2≈1.03972 nats and perplexity≈2.82843. Guided activity B: find a held-out transition where a skewed training model loses to uniform, then explain why this does not make the measurement wrong. Independent check: compute metrics on a different table and reject malformed probabilities.',
        '''# Copy your own stage1–2 functions above these additions.
import numpy as np

def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    raise NotImplementedError

def metrics(ids: list[int], probabilities: np.ndarray) -> dict:
    raise NotImplementedError

def unigram(ids: list[int], size: int, k: float = 1.0) -> np.ndarray:
    raise NotImplementedError''', third,
        [check('known loss and uniform baseline', "p=np.array([[.5,.5],[.25,.75]]); m=metrics([0,1,0],p); assert m['transitions']==2; assert math.isclose(m['cross_entropy'],math.log(8)/2,abs_tol=1e-12,rel_tol=0); assert math.isclose(m['perplexity'],math.sqrt(8),abs_tol=1e-12,rel_tol=0); u=metrics([0,1,0],np.full((2,2),.5)); assert math.isclose(u['cross_entropy'],math.log(2),abs_tol=1e-12,rel_tol=0)"),
         check('unseen pair and smoothing counterexample', "assert math.isinf(metrics([0,1],bigram([0,0],2,0))['cross_entropy']); assert math.isfinite(metrics([0,1],bigram([0,0],2,1))['cross_entropy']); assert metrics([0,1],bigram([0,0,0,0,0,0,1],2))['cross_entropy']>math.log(2)"),
         check('frequency baseline and invalid inputs', "assert np.allclose(unigram([0,0,1],2),[[3/5,2/5],[3/5,2/5]])\nfor ids,p in [([0],np.eye(2)),([0,1],np.ones((2,2))),([0,1],np.array([[1.,0.],[float('nan'),1.]]))]:\n    try: metrics(ids,p)\n    except ValueError: pass\n    else: raise AssertionError('reject undefined metric')")],
        'For held-out 0→1 with assigned probability1/4, loss is ln(4)≈1.38629 nats. Uniform over2 tokens assigns1/2 and has loss ln(2)≈0.69315. The smoothed model loses in this example. Report that result instead of changing the assertion to force a win.',
        ['Count evaluation transitions and write their probabilities.','Sum negative natural logs, divide by transition count, then exponentiate.','Compute uniform and training-only unigram baselines on the identical held-out IDs.','Trace a zero probability and an invalid row rather than hiding them with epsilon.'],
        'Stages1–2; natural logarithm and exp; arithmetic mean; probabilities sum to one. The lesson supplies a hand calculation before code.',
        ['The denominator is the number of adjacent transitions.','Compare measured values even when the more complicated model loses.'])
    add(4, '4 · Choose on development, report on test', 'Produce a reproducible experiment without vocabulary or tuning leakage.',
        'Now connect your tokenizer, model and metrics. The experiment accepts three already separated strings. Fit the vocabulary only on training. Encode development and test with the frozen unknown policy. Before inspecting test metrics, declare positive smoothing candidates in (0,1]. Evaluate each on development and choose the lowest dev cross-entropy; break exact ties with the smaller k. Freeze that choice. Only then evaluate the selected model and both baselines on test. Include train metrics as a diagnostic, never as a replacement for test metrics.\n\nRecord character counts, pair counts, unknown counts, vocabulary, chosen k, the candidate dev losses and your actual execution location. A large unknown fraction is a warning about the chosen corpus/policy, not evidence that the model understands unseen words. Keep examples synthetic or explicitly permitted. The source printed in the lesson is authored; an output is measured only after an explicit host run. Python remains on HOLD in the app.\n\nGuided activity A: compare k=.1 and1 on one declared dev string and freeze your choice in writing. Guided activity B: change test text while leaving train/dev unchanged; your chosen k must stay unchanged. Independent transfer: the checks use a new distribution shift and unseen symbol. Explain whether the unigram baseline captures most of the useful information.\n\n' + EXPERIMENT_REPORT_CONTRACT,
        '''# Carry forward your own tokenizer/bigram/metric functions.
def experiment(training: str, development: str, test: str, candidates: list[float], execution_location: str) -> dict:
    """''' + EXPERIMENT_REPORT_CONTRACT.replace('\n', '\n    ') + '''"""
    raise NotImplementedError''', fourth,
        [check('frozen vocabulary and split accounting', "r=experiment('abababab','abab','abz',[.1,1.],'synthetic host reference test'); assert r['vocabulary']==['<UNK>','a','b']; assert r['characters']=={'train':8,'dev':4,'test':3}; assert r['unknown_characters']['test']==1; assert r['metrics']['test']['transitions']==2; assert r['vocabulary_fit_split']=='train'; assert r['selection_split']=='dev'"),
         check('test data cannot choose smoothing', "a=experiment('aaaaaab','ab','aaaa',[.1,1.],'synthetic host reference test'); b=experiment('aaaaaab','ab','zzzz',[.1,1.],'synthetic host reference test'); assert a['chosen_k']==b['chosen_k']; assert a['dev_candidates']==b['dev_candidates']; assert a['chosen_k']==1.; assert math.isclose(a['baselines']['uniform']['perplexity'],3.,abs_tol=1e-12,rel_tol=0); assert math.isfinite(b['metrics']['test']['cross_entropy'])"),
         check('reproducibility and invalid selection', "a=experiment('ababab','abaa','abba',[1.,.1],'synthetic host reference test'); b=experiment('ababab','abaa','abba',[.1,1.],'synthetic host reference test'); assert a==b\ntry: experiment('abab','ab','ab',[0.],'synthetic host reference test')\nexcept ValueError: pass\nelse: raise AssertionError('declare positive candidates')")],
        'If development chooses k=1, replacing only the test string must not choose k=.1 instead. It can change test loss and unknown count, because those are measured after the choice. That separation is the point of a final test set, not a promise that test will resemble dev.',
        ['Write split roles and candidate k values before running.','Fit and freeze the training-only vocabulary.','Select k with dev only and record tie-breaking.','Score frozen probabilities on test and both baselines; record an honest loss or failure case.'],
        'Stages1–3; dictionaries of measurements; held-out evaluation and parameter selection.',
        ['Test text belongs only after the dev selection step.','Record where code actually ran; never label a source fixture as an iPad run.'])
    add(5, '5 · Sample and inspect a limited model', 'Generate seeded character samples and explain what one-token context cannot do.',
        'At each step, use the row for the current token as a categorical distribution. Draw one next ID, append it, then repeat with the new row. Use a local random generator with a declared seed; do not mutate global random state. Start from a supplied valid token. A sample of count new tokens contains count+1 IDs including the starting token. Convert IDs with your tokenizer; <UNK> should remain visible.\n\nA seed makes a run repeatable within a declared environment, not a guarantee that every RNG library/version produces the same stream. Fluent-looking fragments do not prove long-range structure: this model conditions only on the immediately preceding token. It may memorize local character patterns and emit nonsense.\n\nGuided activity A: step through a deterministic table with one allowed transition per row and predict every output. Guided activity B: compare two mixed rows, then record a seeded sample and explain one failure that loss alone missed. Independent task: preserve zero-probability transitions and validate a different table/seed input.',
        '''# Keep your own earlier functions.
def sample_ids(probabilities: np.ndarray, start: int, count: int, seed: int) -> list[int]:
    raise NotImplementedError''', fifth,
        [check('deterministic path and length', "assert sample_ids(np.array([[0.,1.],[1.,0.]]),0,5,7)==[0,1,0,1,0,1]; assert sample_ids(np.eye(2),1,0,7)==[1]"),
         check('seed repeatability and zero transitions', "p=np.array([[.7,.3],[1.,0.]]); a=sample_ids(p,0,30,19); assert a==sample_ids(p,0,30,19); assert len(a)==31; assert all(p[x,y]>0 for x,y in zip(a,a[1:]))"),
         check('invalid seed token or count', "for start,count in [(-1,2),(2,2),(0,-1)]:\n    try: sample_ids(np.eye(2),start,count,1)\n    except ValueError: pass\n    else: raise AssertionError('validate sampling request')")],
        'For P=[[0,1],[1,0]], starting at0 and requesting three new tokens yields [0,1,0,1] regardless of seed. For a probabilistic row you can predict the distribution, not a particular draw before selecting the generator and seed.',
        ['Inspect the current row and predict allowed next IDs.','Draw one token using a local seeded generator.','Update the current token and repeat, recording the sample length convention.','Compare sample limitations with held-out measurements; neither one proves language understanding.'],
        'Stages1–4; categorical distributions; deterministic seed versus stochastic draw.',
        ['The current token selects the row after every draw.','A zero-probability transition must never be sampled.'])
    add(6, '6 · Diagnose, repair, and document a revision', 'Repair a plausible normalization bug using a fresh counterexample, then recheck the pipeline.',
        'The supplied normalize_counts helper has a deliberate bug: it divides by column totals. Do not fix it by guessing until tests turn green. Draw a fresh count table, label current and next axes, predict row sums and one conditional probability, then inspect the candidate output. Explain the first violated invariant and why a uniform/symmetric example might conceal it. Change the helper to row normalization, preserve the explicit uniform policy for a zero row, and route your bigram function through it.\n\nGuided activity A: use [[1,4],[2,1]] to compare column and row division by hand. Guided activity B: test an unequal-total three-symbol table and a completely empty row. Independent revision: the assessment uses different counts and rechecks the original bigram fixtures. Keep an observed→counterexample→change→recheck note for the final model card. Assistance is a neutral learning fact; a revealed explanation does not substitute for your own revised version passing.',
        '''# Carry forward your own pipeline and route bigram through this helper.
def normalize_counts(counts: np.ndarray) -> np.ndarray:
    # Deliberately wrong: first trace it, then revise your own version.
    return counts / counts.sum(axis=0, keepdims=True)''', sixth,
        [check('new asymmetric debugging table', "c=np.array([[2.,3.,1.],[1.,1.,1.],[4.,0.,2.]]); assert np.allclose(normalize_counts(c),[[1/3,1/2,1/6],[1/3,1/3,1/3],[2/3,0,1/3]],atol=1e-12,rtol=0)")]+bigram_checks()+[
         check('zero row policy and invalid counts', "assert np.allclose(normalize_counts(np.array([[0.,0.],[1.,3.]])),[[.5,.5],[.25,.75]])\ntry: normalize_counts(np.array([[-1.,2.],[1.,0.]]))\nexcept ValueError: pass\nelse: raise AssertionError('negative count')")],
        'For counts [[1,4],[2,1]], row-normalized probabilities are [[1/5,4/5],[2/3,1/3]]. Column-normalizing gives [[1/3,4/5],[2/3,1/5]], whose rows sum to17/15 and13/15. A row-sum check catches this case, but exact asymmetric values also catch normalized tables with the wrong orientation.',
        ['Predict before running the intentionally wrong helper.','Find a small counterexample and show denominators and shapes.','Revise the helper and rerun earlier checks without changing their expectations.','Record what the first failing invariant taught you and what remains unproven.'],
        'Stages1–5; array broadcasting and row-distribution invariants.',
        ['Compute the sum for each row and keep its shape (V,1).','Do not change a correct expected value to match the bug.'])
    seven = add(7, '7 · Write checks that can catch a bug', 'Author behavioral tests rather than copying a green test count.',
        'Now you own the checks. In tests/test_tinylm.py write the three named tests: training vocabulary stays frozen with an unseen symbol; asymmetric transition values match your independent hand count; an unseen in-vocabulary transition has infinite unsmoothed loss. Choose your own small examples and explain expected values before coding assertions. The stage calls your tests against correct functions and five deliberately faulty replacements, including probability transpose and column-normalized counts. Your asymmetric checks must inspect a directional off-diagonal value or a full independently computed table, rather than just one unchanged diagonal entry. Vacuous tests or row-sum-only tests should not pass this review.\n\nThe initial scaffold also has a tests/test_project.py reminder: replace it with your own early tests or consolidate them into test_tinylm.py before final pytest/portfolio checks. Do not leave a deliberately unfinished reminder in your final suite. The CLI harness executes src/main.py and these tests in one namespace; the supplied import guard also supports pytest from the scaffolded repository. Preserve your own implementation in src/main.py. A test count or coverage percentage is only a tool: describe what you tested and what a plausible wrong implementation would still get through. Guided practice: write one test, run it on correct code, deliberately break one behavior and observe the test. Independent task: write the other tests before opening any reference. No test is credited merely because its text is present.',
        '# Carry forward your own complete stage1–6 pipeline here.\n', sixth,
        [check('learner checks run on the actual implementation', 'test_training_vocabulary_is_frozen(); test_asymmetric_transition_values(); test_unseen_transition_has_infinite_unsmoothed_loss()'),
         check('learner checks reject vocabulary, orientation and loss faults', '''
_correct_bigram = bigram
def _column_normalized_bigram(ids, size, k=1):
    c=np.full((size,size),float(k))
    for a,b in zip(ids,ids[1:]): c[a,b]+=1
    totals=c.sum(axis=0,keepdims=True)
    return np.divide(c,totals,out=np.full_like(c,1/size),where=totals>0)
for name, replacement, test in [
    ('build_vocab', lambda training: ['<UNK>']+sorted(set(training) | {next(chr(n) for n in range(0x110000) if not 0xD800 <= n <= 0xDFFF and chr(n) not in training)}), test_training_vocabulary_is_frozen),
    ('bigram', lambda ids,size,k=1: np.eye(size), test_asymmetric_transition_values),
    ('bigram', lambda ids,size,k=1,_b=_correct_bigram: _b(ids,size,k).T, test_asymmetric_transition_values),
    ('bigram', _column_normalized_bigram, test_asymmetric_transition_values),
    ('metrics', lambda ids,p: {'cross_entropy':0.,'perplexity':1.,'transitions':len(ids)-1}, test_unseen_transition_has_infinite_unsmoothed_loss),
]:
    saved=globals()[name]
    try:
        globals()[name]=replacement
        caught=False
        try: test()
        except AssertionError: caught=True
        assert caught, 'Your check must catch this behavioral regression: '+name
    finally:
        globals()[name]=saved
''')],
        'A test that only asserts every row sums to1 accepts an identity matrix, even if the training transitions call for [[3/5,2/5],[1/2,1/2]]. An independent expected entry or full small matrix distinguishes these models. Explain that difference instead of claiming that normalized output is automatically correct.',
        ['Write a prose expectation and choose a non-symmetric example.','Translate that expectation into a learner-authored assertion.','Break the behavior and verify that your assertion notices.','Restore your own implementation and run all checks.'],
        'Stages1–6; assert, pytest naming and the difference between a property and a discriminating example.',
        ['Your examples should catch the behavior, not match implementation text.','A check that accepts every possible output is not evidence.'])
    seven['starterFiles']['tests/test_tinylm.py'] = '''if __name__ != '__main__':
    import math
    import numpy as np
    from main import build_vocab, encode, bigram, metrics

def test_training_vocabulary_is_frozen():
    raise NotImplementedError('Write your own check and explain the expected result')

def test_asymmetric_transition_values():
    raise NotImplementedError('Choose an asymmetric hand-counted example')

def test_unseen_transition_has_infinite_unsmoothed_loss():
    raise NotImplementedError('Check the actual observed zero-probability pair')
'''
    seven['referenceFiles']['tests/test_tinylm.py'] = dedent(LEARNER_TEST_REFERENCE).strip()+'\n'
    artifact_fixture = '''
report=experiment('abababab','abab','abba',[.1,1.],'synthetic host reference test')
fields=['data_origin','data_permission','split_policy','unknown_policy','context_limit','failure_case','revision_note','help_used','execution_location','reproduction_command']
card={name:'Synthetic reference fixture: caller-written evidence' for name in fields}
card.update(metrics=report['metrics'],chosen_k=report['chosen_k'],execution_location=report['execution_location'])
readme='# Synthetic reference artifact\\n\\n'+''.join('## '+heading+'\\nCaller-written synthetic explanation and measured context.\\n\\n' for heading in ['Demo','Quickstart','Results','Decisions','Limits','Revision','Follow-on'])
'''
    add(8, '8 · Publish an honest, reproducible local artifact', 'Export your model card, measured results and learner-written README without overwriting earlier work.',
        'Finish with a useful artifact: your own Python module, your own tests, a fixed synthetic/permitted train/dev/test split, one explicit experiment command, a seeded demo and a model card. State data origin/permission; split roles/counts; frozen training vocabulary and unknown policy; k candidates and dev selection; actual train/dev/test loss and perplexity; uniform/unigram comparison; actual execution location/environment; one failure and one debugging revision; help used; and the one-token context limit. A smooth sample is not evidence of general intelligence. If a baseline wins, explain it.\n\nWrite README sections Demo, Quickstart, Results, Decisions, Limits, Revision and Follow-on in your own words. The export function receives those authored words plus the actual report; it must reject altered metrics and existing files. Export to a new directory, inspect it, then decide what belongs in a public repo. No tool here pushes or opens a PR. Do not include personal notes or another author’s text without permission.\n\nGuided activity A: reproduce a report from a fresh local process with the same splits/settings and compare metrics within declared1e-12 numeric tolerance. Guided activity B: intentionally supply mismatching card metrics or an occupied output directory and verify refusal without changing the old files. Independent capstone: choose a different synthetic corpus, write tests first, explain your dev selection and a failure, then export your own report. The supplied checks run only synthetic temporary-directory examples; they do not certify your corpus, reasoning or independent authorship.\n\nAfter this, attempt the word-level log predictor in FOLLOW-ON.md without a supplied solution. Its token boundaries and malformed records differ from character modeling. Plan new tests and a vocabulary policy before implementing. No reference for that follow-on is bundled.',
        '''# Carry forward your own pipeline; write the README/card yourself.
from pathlib import Path

def export_artifact(directory: Path, report: dict, card: dict, readme: str) -> list[str]:
    raise NotImplementedError''', sixth+EXPORT_REFERENCE,
        [check('export exact measured report in a new directory', artifact_fixture+'''
import tempfile
with tempfile.TemporaryDirectory() as folder:
    out=Path(folder)/'artifact'; names=export_artifact(out,report,card,readme)
    assert set(names)=={'results.json','model-card.json','README.md'}
    assert json.loads((out/'results.json').read_text())==report
    assert json.loads((out/'model-card.json').read_text())['metrics']==report['metrics']
    assert (out/'README.md').read_text()==readme
'''),
         check('existing work and measured values are retained', artifact_fixture+'''
import tempfile
with tempfile.TemporaryDirectory() as folder:
    out=Path(folder); (out/'README.md').write_text('keep my previous writing')
    try: export_artifact(out,report,card,readme)
    except ValueError: pass
    else: raise AssertionError('do not overwrite owner work')
    assert (out/'README.md').read_text()=='keep my previous writing'
    assert not (out/'results.json').exists()
    bad=dict(card); bad['metrics']={'test':{'cross_entropy':0}}
    try: export_artifact(out/'new',report,bad,readme)
    except ValueError: pass
    else: raise AssertionError('do not invent a better result')
''')],
        'A card that says test loss0 when results.json reports a positive loss is inconsistent, even if its prose sounds confident. Keep the measured value and discuss the failure. If artifact/README.md already exists, choose a new export directory instead of replacing your earlier explanation.',
        ['Run your own declared experiment explicitly on an admitted local host.','Write a model card and README that explain the actual report and one revision.','Export into a new directory and verify the saved bytes/metrics.','Attempt the different unsolved follow-on later and record which skills transfer.'],
        'Stages1–7; JSON serialization, pathlib, explicit file paths, retained-file safety and reproducibility.',
        ['The model card must quote the report, not a preferred result.','Preflight every output path before writing any file.'])
    return stages


def taught_series():
    return dict(id='tinylm2', title='Tokenizer → bigram: a taught text-predictor project v2',
        outline=['Training-only tokenizer','Directed smoothed bigrams','Held-out metrics and baselines','Dev selection and frozen test','Seeded sampling and limits','Debug and revise','Learner-written discriminating tests','Model card, README and unsolved follow-on'],
        complete=True, stages=_full_series(),
        resumeBullets=['Only after an actual run: implemented a character model with training-only vocabulary and measured [your test loss] versus [your uniform/unigram losses] on [your declared split].','Diagnosed [your concrete bug] with an asymmetric counterexample and documented [your measured revision result]; no automatic hiring or mastery claim.'],
        demoRecipe='Python on iPad remains HOLD. On an explicitly chosen local host, retain your own src/main.py, tests/test_tinylm.py, synthetic/permitted data and exact environment/settings. Run your own experiment, record its declared execution location and seed, export to a new directory, and link the actual report/sample. Authored lesson examples are not execution receipts.',
        stretchGoals=['Unsolved follow-on: word-level log predictor with different tokenization, malformed-line policy and a new held-out split; no supplied solution.','Investigate sensitivity to k and unknown fraction on declared development data; keep final test frozen.'],
        seniorReviewRubric=['Trace train/dev/test and frozen vocabulary; identify every possible leakage path.','Verify exact directed/overlapping counts and independently derived asymmetric tests.','Review held-out loss/perplexity against both baselines without assuming a winner.','Inspect the learner’s debugging revision and mutation-sensitive authored checks.','Reproduce the final artifact and discuss data permission, help, execution location and one-token limitations.'],
        prerequisites=['Python dictionaries/lists/slicing/functions/exceptions','2D NumPy row/column shapes and broadcasting','Fractions, conditional probability, natural logarithms and means; taught refreshers in stages2–3'],
        authoringStatus='Original source fixtures and optional references. No app execution, learner outcome, mastery or runtime admission is implied.',
        predecessor='gpt (legacy identities preserved, no progress migration)',
        supplementalFiles={
            'MODEL-CARD-TEMPLATE.md':'# Your model card\n\nWrite data origin/permission, exact split counts/roles, training-only vocabulary, unknown policy/counts, selected k and dev candidates, train/dev/test cross-entropy/perplexity, both baselines, sample/seed, actual execution environment, one-token limit, failure/revision evidence, help used and a reproduction command. Quote your measured report; do not fill imaginary results.\n',
            'EXPERIMENT-PLAN.md':'# Plan before running\n\nDeclare train/dev/test strings or permitted synthetic files, boundary indices, unknown policy, k candidates, dev-selection tie rule, seed, environment and execution location. Predict a failure before measuring. Record the frozen plan before examining test metrics. Retain your own code between stages; the CLI archives the prior stage source, and a fresh starter never copies a reference into your work.\n',
            'FOLLOW-ON.md':'# Try without a supplied solution: next-event log predictor\n\nBuild a word-level predictor for original synthetic log lines. Define whether punctuation, empty fields and malformed lines are rejected or tokenized. Freeze vocabulary from training only; choose a policy for unknown words. Compare a declared bigram model to uniform and unigram baselines on a separate test set after development selection. Write checks for token boundaries and malformed records that the character project never handled. Export a report explaining a failure and what you changed. Start with your own plan and tests; no reference solution, worked implementation or promised output is supplied for this follow-on.\n',
        })
