import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from morrowdock_cli import cli

class ProjectsTests(unittest.TestCase):
    def test_every_reference(self):
        for series in cli.catalog():
            for stage in series['stages']:
                with self.subTest(stage=stage['id']), tempfile.TemporaryDirectory() as d:
                    root=Path(d)
                    for name, code in stage['referenceFiles'].items(): (root/name).parent.mkdir(parents=True, exist_ok=True); (root/name).write_text(code, encoding='utf-8')
                    results=cli.run_tests(root,stage)
                    self.assertTrue(results); self.assertTrue(all(ok for _,ok in results),results)
    def test_roundtrip_and_tamper(self):
        key=cli.Ed25519PrivateKey.generate(); doc=cli.envelope({'schema':1},key)
        self.assertEqual(cli.verify(doc),{'schema':1})
        doc['payload']='e30='
        with self.assertRaises(Exception): cli.verify(doc)
    def test_full_series_and_preservation(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); cli.initialize('gpt',root)
            with self.assertRaises(ValueError): cli.advance(root)
            for n, stage in enumerate(cli.catalog()[0]['stages']):
                for name,code in stage['referenceFiles'].items(): (root/name).parent.mkdir(parents=True, exist_ok=True); (root/name).write_text(code, encoding='utf-8')
                self.assertEqual(cli.check(root),0)
                doc=json.loads((root/'morrowdock-progress.json').read_text(encoding='utf-8')); self.assertEqual(len(cli.verify(doc)['completed']),n+1)
                if n==0:
                    (root/'src/main.py').write_text('# changed', encoding='utf-8')
                    with self.assertRaises(ValueError): cli.advance(root)
                    (root/'src/main.py').write_text(stage['referenceFiles']['src/main.py'], encoding='utf-8')
                cli.advance(root)
            self.assertTrue((root/'.morrowdock/stage-1/src/main.py').exists())
            readme=root/'PORTFOLIO-README.md'; readme.write_text('My own write-up', encoding='utf-8'); cli.advance(root); self.assertEqual(readme.read_text(encoding='utf-8'),'My own write-up')
    def test_failed_check_not_completed(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); cli.initialize('agent',root); self.assertEqual(cli.check(root),1)
            self.assertEqual(cli.verify(json.loads((root/'morrowdock-progress.json').read_text(encoding='utf-8')))['completed'],[])
    def test_scaffold_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'src').mkdir(); (root/'src/main.py').write_text('mine', encoding='utf-8')
            with self.assertRaises(ValueError): cli.initialize('gpt',root)
            self.assertEqual((root/'src/main.py').read_text(encoding='utf-8'),'mine')
    def test_timeout_is_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'src').mkdir(); (root/'src/main.py').write_text('pass', encoding='utf-8')
            stage=cli.catalog()[0]['stages'][0]
            with patch('subprocess.run',side_effect=cli.subprocess.TimeoutExpired('test',15)):
                self.assertFalse(any(ok for _,ok in cli.run_tests(root,stage)))
    def test_shared_catalog(self):
        root=Path(__file__).resolve().parents[3]
        native=root/'native-study/Core/Sources/StudyCore/Fixtures/projects-v1.json'
        if not native.exists(): self.skipTest('the app fixture is checked in the private app repository')
        self.assertEqual(json.loads(native.read_text(encoding='utf-8'))['series'],cli.catalog())

if __name__=='__main__': unittest.main()
