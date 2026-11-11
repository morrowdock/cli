"""Independent Fraction oracle and deliberate semantic mutations, not self-goldens."""
from fractions import Fraction
import itertools
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

def stage_two():
    catalog=json.loads((ROOT/'tools/morrowdock-cli/morrowdock_cli/catalog.json').read_text(encoding='utf-8'))
    return next(s for p in catalog['series'] if p['id']=='tinylm2' for s in p['stages'] if s['id']=='tinylm2-2')

def rational_oracle(ids, size, k=Fraction(1)):
    # Scalar conditional-frequency definition; no array broadcasting/transpose path.
    rows=[]
    for current in range(size):
        following=[ids[i+1] for i in range(len(ids)-1) if ids[i]==current]
        denominator=len(following)+k*size
        rows.append([float((following.count(next_id)+k)/denominator) if denominator else 1/size for next_id in range(size)])
    return rows

class BigramQualityTests(unittest.TestCase):
    def test_reference_against_independent_exhaustive_oracle(self):
        scope={}; exec(stage_two()['referenceFiles']['src/main.py'],scope)
        for length in range(7):
            for ids in itertools.product(range(2),repeat=length):
                for k in [Fraction(0),Fraction(1,10),Fraction(1,2),Fraction(1)]:
                    with self.subTest(ids=ids,k=k):
                        self.assertTrue(np.allclose(scope['bigram'](list(ids),2,float(k)),rational_oracle(ids,2,k),atol=1e-12,rtol=0))
    def test_audit_counterexamples_and_hand_count(self):
        self.assertEqual(rational_oracle([0,0,1],2),[[.5,.5],[.5,.5]])
        self.assertEqual(rational_oracle([0,1,1],2),[[1/3,2/3],[1/3,2/3]])
        self.assertEqual(rational_oracle([0,0,0,1,1,0],2),[[3/5,2/5],[1/2,1/2]])
    def test_actual_stage_checks_reject_all_four_mutations(self):
        stage=stage_two(); checks=stage['visibleTests']+stage['hiddenTests']
        scope={}; exec(stage['referenceFiles']['src/main.py'],scope)
        for check in checks: exec(check['source'],scope)
        for mutation in ['transpose','columns','disjoint','missing_keepdims']:
            def wrong(ids,size,k=1,mode=mutation):
                counts=np.full((size,size),float(k))
                pairs=zip(ids[::2],ids[1::2]) if mode=='disjoint' else zip(ids,ids[1:])
                for a,b in pairs: counts[a,b]+=1
                axis=0 if mode=='columns' else 1
                p=counts/counts.sum(axis=axis,keepdims=mode!='missing_keepdims')
                return p.T if mode=='transpose' else p
            rejected=[]
            for check in checks:
                try: exec(check['source'],{'bigram':wrong,'np':np})
                except AssertionError: rejected.append(check['name'])
            with self.subTest(mutation=mutation): self.assertTrue(rejected,mutation)
    def test_nondefault_hand_counts_against_independent_oracle(self):
        scope={}; exec(stage_two()['referenceFiles']['src/main.py'],scope)
        for ids,size,k,expected in [
            ([0,0,0,1,1,0],2,Fraction(1,2),[[5/8,3/8],[1/2,1/2]]),
            ([0,0,1],3,Fraction(0),[[1/2,1/2,0],[1/3,1/3,1/3],[1/3,1/3,1/3]]),
        ]:
            self.assertTrue(np.allclose(rational_oracle(ids,size,k),expected,atol=1e-12,rtol=0))
            actual=scope['bigram'](ids,size,float(k))
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue(np.allclose(actual,expected,atol=1e-12,rtol=0))
    def test_actual_stage_checks_reject_ignored_k_and_nan_fallback(self):
        stage=stage_two(); scope={}; exec(stage['referenceFiles']['src/main.py'],scope)
        correct=scope['bigram']
        def ignored_k(ids,size,k=1): return correct(ids,size,1.)
        def nan_fallback(ids,size,k=1):
            c=np.full((size,size),float(k))
            for a,b in zip(ids,ids[1:]): c[a,b]+=1
            with np.errstate(divide='ignore',invalid='ignore'):
                return c/c.sum(axis=1,keepdims=True)
        for mutant in [ignored_k,nan_fallback]:
            rejected=[]
            for check in stage['visibleTests']+stage['hiddenTests']:
                try: exec(check['source'],{'bigram':mutant,'np':np})
                except AssertionError: rejected.append(check['name'])
            self.assertTrue(rejected,mutant.__name__)
    def test_smoothing_is_not_a_universal_improvement(self):
        scope={}; exec(stage_two()['referenceFiles']['src/main.py'],scope)
        p=scope['bigram']([0,0,0,0,0,0,1],2)
        self.assertGreater(-np.log(p[0,1]),np.log(2))

if __name__=='__main__': unittest.main()
