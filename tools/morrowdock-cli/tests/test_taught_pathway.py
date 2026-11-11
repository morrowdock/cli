"""Host reference/authoring checks only: no learner execution or runtime admission."""
import json
import ast
import contextlib
import io
import sys
import tempfile
import math
from pathlib import Path
import hashlib
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
CATALOG=ROOT/'tools/morrowdock-cli/morrowdock_cli/catalog.json'
sys.path.insert(0,str(ROOT/'tools/morrowdock-cli'))
from morrowdock_cli import cli

def pathway():
    return next(p for p in json.loads(CATALOG.read_text(encoding='utf-8'))['series'] if p['id']=='tinylm2')

def reference_scope(stage):
    scope={'__name__':'__main__'}
    for name,source in stage['referenceFiles'].items():
        exec(compile(source,name,'exec'),scope)
    return scope

class TaughtPathwayTests(unittest.TestCase):
    def test_every_authored_reference_passes_its_actual_checks(self):
        project=pathway()
        self.assertEqual(len(project['stages']),8)
        for stage in project['stages']:
            for check in stage['visibleTests']+stage['hiddenTests']:
                with self.subTest(stage=stage['id'],check=check['name']):
                    scope=reference_scope(stage)
                    exec(compile(check['source'],check['name'],'exec'),scope)
    def test_frozen_legacy_history_and_new_identity(self):
        baseline=json.loads((Path(__file__).parent/'fixtures/projects-094413d2-manifest.json').read_text(encoding='utf-8'))
        actual=json.loads(CATALOG.read_text(encoding='utf-8'))
        self.assertEqual([p['id'] for p in actual['series'][:6]],[p['id'] for p in baseline['series']])
        for project, expected in zip(actual['series'][:6],baseline['series']):
            digest=hashlib.sha256(json.dumps(project,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            self.assertEqual(digest,expected['sha256'],project['id'])
            self.assertEqual([s['id'] for s in project['stages']],expected['stageIDs'])
        self.assertEqual([s['id'] for s in pathway()['stages']],[f'tinylm2-{n}' for n in range(1,9)])
        native=ROOT/'native-study/Core/Sources/StudyCore/Fixtures/projects-v1.json'
        if native.exists(): self.assertEqual(json.loads(native.read_text(encoding='utf-8')),actual)  # app fixture: private app repository only
    def test_dev_selection_and_metrics_against_scalar_probabilities(self):
        scope=reference_scope(pathway()['stages'][3])
        report=scope['experiment']('aaaaaab','ab','ab',[.1,1.],'synthetic host reference test')
        # Training vocabulary is UNK,a,b. a->a occurs five times, a->b once.
        # With add1: a row is [1/9,6/9,2/9]. Held-out ab has one pair.
        self.assertEqual(report['chosen_k'],1.)
        self.assertTrue(math.isclose(report['metrics']['test']['cross_entropy'],-math.log(2/9),abs_tol=1e-12,rel_tol=0))
        self.assertGreater(report['metrics']['test']['cross_entropy'],report['baselines']['uniform']['cross_entropy'])
        self.assertEqual(report['vocabulary'],['<UNK>','a','b'])
        changed=scope['experiment']('aaaaaab','ab','zzzz',[.1,1.],'synthetic host reference test')
        self.assertEqual(changed['chosen_k'],report['chosen_k'])
        self.assertEqual(changed['dev_candidates'],report['dev_candidates'])
        self.assertEqual(changed['unknown_characters']['test'],4)
    def test_learners_tests_actually_discriminate_and_starters_are_unsolved(self):
        stage=pathway()['stages'][6]
        scope=reference_scope(stage)
        for check in stage['visibleTests']+stage['hiddenTests']: exec(check['source'],scope)
        self.assertIn('raise NotImplementedError',stage['starterFiles']['tests/test_tinylm.py'])
        self.assertNotEqual(stage['starterFiles']['tests/test_tinylm.py'],stage['referenceFiles']['tests/test_tinylm.py'])
        debugging=pathway()['stages'][5]
        scope={'np':np}; exec(debugging['starterFiles']['src/main.py'],scope)
        with self.assertRaises(AssertionError): exec(debugging['visibleTests'][0]['source'],scope)
    def test_stage_seven_accepts_learner_vocabulary_containing_z(self):
        stage=pathway()['stages'][6]
        scope=reference_scope(stage)
        exec("""def test_training_vocabulary_is_frozen():
    vocab=build_vocab('za')
    assert vocab==['<UNK>','a','z']
    before=list(vocab)
    assert encode('?',vocab)==[0]
    assert vocab==before
""",scope)
        for check in stage['visibleTests']+stage['hiddenTests']:
            exec(check['source'],scope)

    def test_stage_one_requires_transfer_beyond_guided_inputs(self):
        stage=pathway()['stages'][0]
        scope=reference_scope(stage)
        original=scope['split_text']
        # A learner who special-cases only the guided trace cannot pass transfer.
        scope['split_text']=lambda text,a,b: original(text,a,b) if text=='ab ab\nzz' else ('wrong','wrong','wrong')
        transfer=next(c for c in stage['visibleTests']+stage['hiddenTests']
                      if c['name']=='independent Unicode split and frozen vocabulary')
        with self.assertRaises(AssertionError):
            exec(transfer['source'],scope)

    def test_stage_seven_rejects_orientation_blind_single_entry_check(self):
        stage=pathway()['stages'][6]; scope=reference_scope(stage)
        exec("""def test_asymmetric_transition_values():
    assert abs(bigram([0,0,0,1,1,0],2)[0,0]-3/5)<1e-12
""",scope)
        exec(stage['visibleTests'][0]['source'],scope)
        with self.assertRaises(AssertionError):
            exec(stage['hiddenTests'][0]['source'],scope)
    def test_stage_four_teaches_exact_output_schema_and_candidate_order(self):
        stage=pathway()['stages'][3]
        function=next(node for node in ast.parse(stage['starterFiles']['src/main.py']).body if isinstance(node,ast.FunctionDef) and node.name=='experiment')
        contract=ast.get_docstring(function)
        self.assertIsNotNone(contract)
        scope=reference_scope(stage)
        a=scope['experiment']('ababab','abaa','abba',[1.,.1,1.],'synthetic host reference test')
        b=scope['experiment']('ababab','abaa','abba',[.1,1.],'synthetic host reference test')
        self.assertEqual(a,b)
        self.assertEqual([row['k'] for row in a['dev_candidates']],[.1,1.])
        for key in a:
            self.assertIn('"'+key+'"',contract)
            self.assertIn('"'+key+'"',stage['lesson'])
        for key in ['cross_entropy','perplexity','transitions']:
            self.assertIn('"'+key+'"',contract)
        self.assertIn('sort them ascending',contract)
        self.assertEqual([len(s['visibleTests'])+len(s['hiddenTests']) for s in pathway()['stages']],[4,4,3,3,3,6,2,2])
    def test_full_teaching_and_no_solution_follow_on(self):
        project=pathway()
        for stage in project['stages']:
            self.assertTrue(stage['prerequisites'])
            self.assertGreater(len(stage['lesson']),650)
            self.assertGreater(len(stage['workedExample']),100)
            self.assertGreaterEqual(len(stage['walkthrough']),4)
            self.assertTrue(stage['guidedPrompt'])
            self.assertTrue(stage['visibleTests'] and stage['hiddenTests'])
        follow_on=project['supplementalFiles']['FOLLOW-ON.md']
        self.assertIn('word-level',follow_on)
        self.assertNotIn('def ',follow_on)
        self.assertNotIn('```python',follow_on)
        self.assertNotIn('optionalRevealFiles',project)
        self.assertIn('HOLD',project['demoRecipe'])
        self.assertIn('actual execution',project['stages'][-1]['lesson'])

class TaughtCLIIntegrationTests(unittest.TestCase):
    def test_real_cli_progresses_all_new_stages_without_replacing_work_on_reveal(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder)
            cli.initialize('tinylm2',root)
            self.assertTrue((root/'FOLLOW-ON.md').exists())
            for number,stage in enumerate(pathway()['stages'],1):
                self.assertEqual(cli.check(root),1,'a fresh learner starter must not be a supplied completed solution')
                before={name:(root/name).read_bytes() for name in stage['starterFiles']}
                cli.reveal(root)
                self.assertEqual(before,{name:(root/name).read_bytes() for name in stage['starterFiles']})
                for name,source in stage['referenceFiles'].items():
                    (root/name).parent.mkdir(parents=True,exist_ok=True)
                    (root/name).write_text(source, encoding='utf-8')
                self.assertEqual(cli.check(root),0,stage['id'])
                envelope=json.loads((root/'morrowdock-progress.json').read_text(encoding='utf-8'))
                progress=cli.verify(envelope)
                self.assertEqual([item['stageID'] for item in progress['completed']],[f'tinylm2-{n}' for n in range(1,number+1)])
                self.assertIn(stage['id'],progress['practicedWithHelp'])
                cli.advance(root)
            self.assertTrue((root/'.morrowdock/stage-7/tests/test_tinylm.py').exists())
            self.assertTrue((root/'PORTFOLIO-README.md').exists())
            self.assertEqual((root/'FOLLOW-ON.md').read_text(encoding='utf-8'),pathway()['supplementalFiles']['FOLLOW-ON.md'])

if __name__=='__main__': unittest.main()
