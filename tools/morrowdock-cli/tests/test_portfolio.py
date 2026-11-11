import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from morrowdock_cli import cli, portfolio

class LearningTests(unittest.TestCase):
    def test_teaching_and_optional_reveal_preserve_own_work(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); cli.initialize('agent',root); own=(root/'src/main.py').read_bytes()
            before=(root/'.morrowdock/state.json').read_bytes()
            cli.learn(root)
            self.assertEqual((root/'.morrowdock/state.json').read_bytes(),before)
            with self.assertRaises(ValueError): cli.reveal(root)
            cli.learn(root,True); self.assertEqual(cli.check(root),1); cli.reveal(root)
            self.assertEqual((root/'src/main.py').read_bytes(),own)
            state=json.loads((root/'.morrowdock/state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['completed'],[])
            self.assertEqual(state['practicedWithHelp'],['agent-1'])
            stage=next(s for s in cli.catalog() if s['id']=='agent')['stages'][0]
            (root/'src/main.py').write_text(stage['referenceFiles']['src/main.py'], encoding='utf-8')
            self.assertEqual(cli.check(root),0)
            completed=json.loads((root/'.morrowdock/state.json').read_text(encoding='utf-8'))['completed']
            (root/'src/main.py').write_bytes(own); self.assertEqual(cli.check(root),1)
            self.assertEqual(json.loads((root/'.morrowdock/state.json').read_text(encoding='utf-8'))['completed'],completed)
            with self.assertRaises(ValueError): cli.advance(root)
    def test_starters_have_structure_and_do_not_ship_full_solutions(self):
        for series in cli.catalog():
            with self.subTest(series=series['id']), tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                root=Path(d); cli.initialize(series['id'],root)
                for name in ['src','tests','docs','.github/workflows/ci.yml','README.md','LICENSE']:
                    self.assertTrue((root/name).exists(),name)
                self.assertFalse((root/'torch_path.py').exists())
                for stage in series['stages']:
                    self.assertTrue(stage['walkthrough']); self.assertTrue(stage['workedExample'])
                self.assertTrue(series['demoRecipe']); self.assertTrue(series['seniorReviewRubric'])
    def test_unfinished_starter_is_not_portfolio_ready(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); cli.initialize('gpt',root)
            state,series,_=cli.load(root)
            checks=dict(portfolio.check(root,series,state))
            self.assertFalse(checks['all released stages completed'])
            self.assertFalse(checks['README completed with evidence'])
            self.assertFalse(checks['senior review evidence recorded'])
            self.assertFalse(checks['annotated release tag at HEAD'])

class PortfolioEvidenceTests(unittest.TestCase):
    def test_real_local_tools_and_git_evidence(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); cli.initialize('gpt',root); state,series,_=cli.load(root)
            # Synthetic completed-series fixture; reference correctness is tested separately.
            state['completed']=[{'stageID':s['id']} for s in series['stages']]
            (root/'src/main.py').write_text('def twice(value: int) -> int:\n    return value * 2\n', encoding='utf-8')
            (root/'tests/test_project.py').write_text('from main import twice\n\n\ndef test_positive_and_negative() -> None:\n    assert twice(3) == 6\n    assert twice(-2) == -4\n', encoding='utf-8')
            readme='# Synthetic artifact\n\nA small typed arithmetic fixture used only by this test.\n\n![CI](https://github.com/fixture/example/actions/workflows/ci.yml/badge.svg)\n![coverage](https://img.shields.io/badge/coverage-100-blue)\n'
            for name in ['Demo','Quickstart','Architecture','Design decisions','Results','Lessons','Next steps','Learning notes']:
                readme+='\n## '+name+'\nA synthetic fixture with explicit reproducible evidence and documented limitations.\n'
                if name=='Architecture': readme+='```mermaid\nflowchart LR\nInput --> Core\nCore --> Result\n```\n'
                if name=='Demo': readme+='docs/demo.html\n'
            (root/'README.md').write_text(readme, encoding='utf-8'); (root/'docs/demo.html').write_text('<p>Synthetic test demo</p>', encoding='utf-8')
            for name in ['LICENSE','CHANGELOG.md','docs/ROADMAP.md','docs/REVIEW.md','docs/BLOG-DRAFT.md','docs/RESUME.md','docs/DEMO.md']:
                text=(root/name).read_text(encoding='utf-8').replace('TODO_YEAR','2026').replace('TODO_OWNER','Synthetic Fixture').replace('TODO:', 'Synthetic test evidence:').replace('TODO','Synthetic').replace('[measured','[recorded').replace('- [ ]','- [x]' if name=='docs/REVIEW.md' else '- [ ]')
                if name=='CHANGELOG.md': text+='\n## v0.1.0\nSynthetic fixture release; typed arithmetic, positive and negative checks, no product claims.\n'
                (root/name).write_text(text, encoding='utf-8')
            (root/'docs/benchmarks.json').write_text(json.dumps({'measurements':[{'metric':'test count','value':2,'unit':'cases','command':'python -m pytest','environment':'synthetic local fixture','baseline':'one positive case'}]}), encoding='utf-8')
            def git(*args): return subprocess.run(['git',*args],cwd=root,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            git('init'); git('config','user.email','fixture@example.invalid'); git('config','user.name','Synthetic Fixture')
            for i in range(12):
                (root/'docs/history.txt').write_text(str(i), encoding='utf-8'); git('add','.'); git('commit','-m',f'Stage {i//2+1}: document focused synthetic step {i}')
            git('tag','-a','v0.1.0','-m','Synthetic verified fixture release')
            checks=portfolio.check(root,series,state)
            self.assertTrue(all(ok for _,ok in checks),[name for name,ok in checks if not ok])
            (root/'src/main.py').write_text('def twice(value: int) -> int:\n    return value * 3\n', encoding='utf-8')
            checks=dict(portfolio.check(root,series,state))
            self.assertFalse(checks['tests with at least 80% line and branch coverage'])
            self.assertFalse(checks['clean tracked worktree'])

if __name__=='__main__': unittest.main()
