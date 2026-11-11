"""Host authoring checks for the taught GPT series (gpt2). No learner execution or runtime admission.

Every reference passes every check; every named wrong implementation is rejected by at least one
hidden check; starters are unsolved; the catalog copies match the generator.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / 'tools/morrowdock-cli'
CATALOG = CLI / 'morrowdock_cli/catalog.json'
sys.path.insert(0, str(CLI))
import projects_gpt_v2  # noqa: E402
import projects_shell_v2  # noqa: E402


def series(series_id='gpt2'):
    return next(p for p in json.loads(CATALOG.read_text(encoding='utf-8'))['series'] if p['id'] == series_id)


def run(source_files, check, override=None):
    scope = {'__name__': '__main__'}
    for name, source in source_files.items():
        exec(compile(source, name, 'exec'), scope)
    if override:
        exec(compile(override, 'mutant', 'exec'), scope)
    exec(compile(check['source'], check['name'], 'exec'), scope)


class TaughtGPTTests(unittest.TestCase):
    SERIES, MODULE, BUILD, COUNT = 'gpt2', projects_gpt_v2, 'taught_gpt_series', 6

    def test_catalog_matches_generator_and_native_copy(self):
        built = getattr(self.MODULE, self.BUILD)()
        self.assertEqual(series(self.SERIES), json.loads(json.dumps(built, ensure_ascii=False)))
        self.assertEqual([s['id'] for s in series(self.SERIES)['stages']], [f'{self.SERIES}-{n}' for n in range(1, self.COUNT + 1)])
        native = ROOT / 'native-study/Core/Sources/StudyCore/Fixtures/projects-v1.json'
        if native.exists():  # the app fixture exists only in the private app repository
            self.assertEqual(json.loads(native.read_text(encoding='utf-8')), json.loads(CATALOG.read_text(encoding='utf-8')))

    def test_every_reference_passes_every_check(self):
        for stage in series(self.SERIES)['stages']:
            for check in stage['visibleTests'] + stage['hiddenTests']:
                with self.subTest(stage=stage['id'], check=check['name']):
                    run(stage['referenceFiles'], check)

    def test_every_wrong_implementation_is_rejected_by_a_hidden_check(self):
        for stage in series(self.SERIES)['stages']:
            for name, _function, source in self.MODULE.MUTANTS[stage['id']]:
                self.assertNotIn(source.strip(), stage['referenceFiles']['src/main.py'], f'{name} is identical to the reference')
                caught = []
                for check in stage['hiddenTests']:
                    try:
                        run(stage['referenceFiles'], check, source)
                    except Exception:
                        caught.append(check['name'])
                with self.subTest(stage=stage['id'], mutant=name):
                    self.assertTrue(caught, f'{name} passes every hidden check')

    def test_starters_are_unsolved_and_lessons_are_taught(self):
        for stage in series(self.SERIES)['stages']:
            src = stage['starterFiles']['src/main.py']
            self.assertIn('raise NotImplementedError', src)
            for check in stage['visibleTests'] + stage['hiddenTests']:
                with self.subTest(stage=stage['id'], check=check['name']):
                    with self.assertRaises(Exception):
                        run(stage['starterFiles'], check)
            lesson = stage['lesson']
            for marker in ['Prerequisites:', 'Misconception', 'Guided activity A', 'Guided activity B', 'On your own']:
                self.assertIn(marker, lesson, stage['id'])
            self.assertGreaterEqual(len(stage['hiddenTests']), 3)
            self.assertGreaterEqual(len(self.MODULE.MUTANTS[stage['id']]), 3)

    def test_native_projects_tests_match_catalog(self):
        """ProjectsTests.swift hard-codes series ids, stage counts and host-process stages; keep them in step."""
        import re
        swift_path = ROOT / 'native-study/Core/Tests/StudyCoreTests/ProjectsTests.swift'
        if not swift_path.exists():
            self.skipTest('ProjectsTests.swift is checked in the private app repository')
        swift = swift_path.read_text(encoding='utf-8')
        catalog = json.loads(CATALOG.read_text(encoding='utf-8'))['series']
        ids = re.search(r'map\(\\\.id\), \[([^\]]*)\]\)', swift).group(1)
        counts = re.search(r'stages\.count \}, \[([^\]]*)\]\)', swift).group(1)
        host = re.search(r'requiresHostProcess == true \}\.map\(\\\.id\), \[([^\]]*)\]\)', swift).group(1)
        self.assertEqual(re.findall(r'"([^"]+)"', ids), [s['id'] for s in catalog])
        self.assertEqual([int(n) for n in counts.split(',')], [len(s['stages']) for s in catalog])
        self.assertEqual(re.findall(r'"([^"]+)"', host), [st['id'] for s in catalog for st in s['stages'] if st.get('requiresHostProcess')])

    def test_portfolio_templates_and_no_invented_claims(self):
        s = series(self.SERIES)
        files = s['supplementalFiles']
        self.assertIn('## Results (measured', files['README-TEMPLATE.md'])
        self.assertIn('No reference implementation', files['FOLLOW-ON.md'])
        text = json.dumps(s).lower()
        for banned in ['mastered', 'mastery achieved', 'hired', 'guaranteed', 'state-of-the-art']:
            self.assertNotIn(banned, text)


class SamplerTemperatureTests(unittest.TestCase):
    """Temperature divides the logits: a deterministic probability probe at T = 0.5 and T = 2."""

    def setUp(self):
        self.stage = next(s for s in series('gpt2')['stages'] if s['id'] == 'gpt2-6')
        self.check = next(c for c in self.stage['hiddenTests'] if c['name'].startswith('sampler rules'))

    def test_reference_probabilities_match_an_independent_closed_form(self):
        import math
        import numpy as np
        scope = {'__name__': '__main__'}
        exec(compile(self.stage['referenceFiles']['src/main.py'], 'ref', 'exec'), scope)
        for t, logit_gap in [(0.5, 4.0), (2.0, 1.0)]:
            seen = []

            class Recorder:
                def choice(self, a, size=None, replace=True, p=None):
                    seen.append(list(p))
                    return 0
            scope['sample_next'](np.array([0.0, 2.0]), t, 2, Recorder())
            expected = 1 / (1 + math.exp(-logit_gap))
            self.assertAlmostEqual(max(seen[-1]), expected, places=12)
            self.assertAlmostEqual(min(seen[-1]), 1 - expected, places=12)

    def test_wrong_temperature_rules_fail_the_sampler_check(self):
        authored = [(n, s) for n, _f, s in projects_gpt_v2.MUTANTS['gpt2-6'] if n.startswith('temperature multiplies')]
        self.assertEqual(len(authored), 2)
        ignored = '''
def sample_next(logits_row, temperature, top_k, rng):
    if temperature < 0 or top_k < 1: raise ValueError('bad')
    row = np.asarray(logits_row, dtype=float)
    if temperature == 0: return int(row.argmax())
    keep = np.argsort(-row, kind='stable')[:top_k]
    return int(keep[rng.choice(len(keep), p=softmax(row[keep]))])
'''
        for name, source in authored + [('temperature ignored (test-only)', ignored)]:
            with self.subTest(mutant=name):
                with self.assertRaisesRegex(AssertionError, r'T = (0\.5|2\.0)'):
                    run(self.stage['referenceFiles'], self.check, source)

    def test_correct_samplers_with_other_rng_usage_pass(self):
        """Correct samplers that make internal choice calls or normalise the Generator must pass."""
        head = '''
def sample_next(logits_row, temperature, top_k, rng):
    if temperature < 0 or top_k < 1: raise ValueError('bad')
    row = np.asarray(logits_row, dtype=float)
    if temperature == 0: return int(row.argmax())
    keep = np.argsort(-row, kind='stable')[:top_k]; probs = softmax(row[keep] / temperature)
'''
        bodies = {
            'rejection sampling with a uniform choice proposal': '''
    while True:
        i = int(rng.choice(len(keep), p=np.full(len(keep), 1 / len(keep))))
        if rng.random() < probs[i] / probs.max(): return int(keep[i])
''',
            'Generator normalised with default_rng(rng)': '''
    rng = np.random.default_rng(rng)
    return int(rng.choice(keep, p=probs))
''',
            'inverse CDF with rng.random': '''
    i = int(np.searchsorted(np.cumsum(probs), rng.random(), side='right'))
    return int(keep[min(i, len(keep) - 1)])
''',
            'positional p and size=1': '''
    return int(rng.choice(keep, 1, True, probs)[0])
''',
        }
        for name, body in bodies.items():
            with self.subTest(sampler=name):
                run(self.stage['referenceFiles'], self.check, head + body)


class TaughtShellTests(TaughtGPTTests):
    SERIES, MODULE, BUILD, COUNT = 'shell2', projects_shell_v2, 'taught_shell_series', 4


if __name__ == '__main__':
    unittest.main()
