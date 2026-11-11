"""Build original shared curriculum. No private notes or external course text."""
import json
from pathlib import Path
from textwrap import dedent

series = []
def project(id, title, outline, complete, bullets):
    p = dict(id=id,title=title,outline=outline,complete=complete,resumeBullets=bullets,stages=[])
    series.append(p); return p

def stage(p, title, goal, lesson, signature, solution, visible, hidden, hints, language='python', extra=None):
    n = len(p['stages']) + 1
    source = dedent(solution).strip() + '\n'
    stub = dedent(signature).strip() + '\n'
    ext = 'lua' if language == 'lua' else 'py'
    item = dict(id=f"{p['id']}-{n}",title=title,goal=goal,lessonRef=f"projects/{p['id']}/{n}",lesson=lesson,
        language=language,starterFiles={f'main.{ext}':stub},referenceFiles={f'main.{ext}':source},
        visibleTests=[dict(name=name,source=dedent(code).strip()) for name,code in visible],
        hiddenTests=[dict(name=name,source=dedent(code).strip()) for name,code in hidden],hints=hints,
        reviewRubric=['Explain each input and output in your own words.','Show one boundary case and why it matters.','Compare your approach to a simpler baseline.','Describe the runtime cost and one limitation.'])
    if extra: item.update(extra)
    p['stages'].append(item)

gpt=project('gpt','Build your own GPT',['Tokenizer','Bigram','Training loop','MLP','Self-attention','Tiny transformer'],True,['Built a NumPy language-model pipeline from character tokens to a causal transformer; documented deterministic correctness checks.','Compared count-based and learned next-token models and recorded loss, sampling behavior, and limitations.'])
stage(gpt,'1 · Give text an alphabet','Build a reversible character tokenizer.','A token is a number standing for a symbol. Start with sorted unique characters so two runs agree. Encoding looks up each character; decoding joins characters by index. Whitespace is data too. A character tokenizer is intentionally simpler than production subword tokenization. Try a short Lua function as text. Predict the IDs before running the visible example.', '''def vocabulary(text):
    raise NotImplementedError

def encode(text, vocab):
    raise NotImplementedError

def decode(ids, vocab):
    raise NotImplementedError''','''
def vocabulary(text):
    return sorted(set(text))
def encode(text, vocab):
    lookup = {c:i for i,c in enumerate(vocab)}
    return [lookup[c] for c in text]
def decode(ids, vocab):
    if any(i < 0 or i >= len(vocab) for i in ids): raise ValueError('Unknown token')
    return ''.join(vocab[i] for i in ids)
''',[('round trip',"v = vocabulary('cab c'); assert decode(encode('cab c', v), v) == 'cab c'")],[('unicode and empty',"assert vocabulary('') == []; v=vocabulary('λ 🚁λ'); assert decode(encode('λ 🚁λ',v),v)=='λ 🚁λ'"),('unknown token',"try: decode([-1], ['a'])\nexcept ValueError: pass\nelse: raise AssertionError('reject negative ids')")],['Build a character-to-index dictionary once.','Do not strip spaces or newlines.'])
stage(gpt,'2 · Count what follows','Estimate next-token probabilities with smoothing.','Each row represents the current token; each column is a possible next token. Count adjacent pairs, add one to every cell, then divide each row by its sum. Smoothing gives unseen transitions a chance. This model sees one token of context. Compare a repeated phrase with shuffled text: what information survives?', '''import numpy as np

def bigram(ids, size):
    # Return a size x size probability matrix with add-one smoothing.
    raise NotImplementedError''','''
import numpy as np
def bigram(ids, size):
    counts=np.ones((size,size), dtype=float)
    for a,b in zip(ids,ids[1:]): counts[a,b]+=1
    return counts/counts.sum(axis=1,keepdims=True)
''',[('counts become probabilities',"p=bigram([0,1,0],2); assert np.allclose(p, [[1/3,2/3],[2/3,1/3]])")],[('empty and unseen row',"p=bigram([],3); assert np.allclose(p,1/3); assert np.allclose(bigram([0,0],3).sum(1),1)")],['Use keepdims=True when normalizing rows.','Count overlapping pairs, not disjoint pairs.'])
stage(gpt,'3 · Learn instead of count','Train bigram logits by gradient descent and record loss.','Softmax converts scores to probabilities. Cross entropy is minus the log probability assigned to the observed next token. Its gradient for one row is predicted probabilities minus a one-hot target. Average over pairs, then subtract learning-rate times gradient. Save a loss before and after training. A lower training loss alone says nothing about unseen text; keep a held-out string for your write-up.', '''import numpy as np

def train(ids, size, steps=80, lr=0.5):
    # Return (logits, loss_history), including initial loss.
    raise NotImplementedError''','''
import numpy as np
def train(ids,size,steps=80,lr=0.5):
    if len(ids)<2: raise ValueError('Need two tokens')
    x=np.asarray(ids[:-1]); y=np.asarray(ids[1:]); w=np.zeros((size,size)); history=[]
    for step in range(steps+1):
        z=w[x]; p=np.exp(z-z.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
        history.append(float(-np.log(p[np.arange(len(y)),y]).mean()))
        if step==steps: break
        g=p.copy(); g[np.arange(len(y)),y]-=1; grad=np.zeros_like(w)
        np.add.at(grad,x,g/len(y)); w-=lr*grad
    return w,history
''',[('loss falls',"w,h=train([0,1]*20,2); assert h[-1]<h[0]*0.3; assert len(h)==81")],[('zero steps and input guard',"w,h=train([0,1],2,steps=0); assert np.allclose(w,0); assert np.isclose(h[0],np.log(2))\ntry: train([0],2)\nexcept ValueError: pass\nelse: raise AssertionError('need pairs')")],['Use a separate gradient array; repeated input rows must accumulate.','Subtract the maximum score before exponentiating.'])
stage(gpt,'4 · Add a hidden layer','Backpropagate through a tanh MLP on one-hot contexts.','A hidden layer can learn features rather than memorize a single score table. Compute H=tanh(XW1), scores=HW2. Backpropagate from softmax through W2, then multiply by (1-H²) to pass through tanh. Initialize with a fixed seed. This first MLP consumes one-hot contexts; extend it to concatenated context embeddings after the checks pass.', '''import numpy as np

def train_mlp(x, y, classes, steps=250, lr=0.3):
    # x is a 2D feature matrix, y integer labels. Return w1,w2,losses.
    raise NotImplementedError''','''
import numpy as np
def train_mlp(x,y,classes,steps=250,lr=0.3):
    x=np.asarray(x,dtype=float); y=np.asarray(y); rng=np.random.default_rng(7)
    a=rng.normal(0,0.2,(x.shape[1],12)); b=rng.normal(0,0.2,(12,classes)); losses=[]
    for i in range(steps+1):
        h=np.tanh(x@a); z=h@b; p=np.exp(z-z.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
        losses.append(float(-np.log(p[np.arange(len(y)),y]).mean()))
        if i==steps: break
        dz=p.copy(); dz[np.arange(len(y)),y]-=1; dz/=len(y)
        da=x.T@((dz@b.T)*(1-h*h)); db=h.T@dz
        a-=lr*da; b-=lr*db
    return a,b,losses
''',[('learn a token mapping',"a,b,h=train_mlp(np.eye(3),[1,2,0],3); assert h[-1]<0.1; assert (np.tanh(np.eye(3)@a)@b).argmax(1).tolist()==[1,2,0]")],[('deterministic and shaped',"a,b,h=train_mlp(np.eye(2),[1,0],2); aa,bb,hh=train_mlp(np.eye(2),[1,0],2); assert np.array_equal(a,aa); assert b.shape==(12,2); assert h[-1]<h[0]")],['Compute both gradients before changing either weight.','One row of x is one training example.'])
stage(gpt,'5 · Look back with attention','Implement scaled, causal self-attention.','Queries ask what a position needs; keys describe what earlier positions offer; values carry the information. Scores are QKᵀ divided by the square root of key width. Mask future columns before softmax. Each row of the resulting attention matrix sums to one. Change only a future token: earlier outputs must stay unchanged. That is the core causal test.', '''import numpy as np

def attention(q,k,v):
    # Return (weighted_values, causal_attention_weights).
    raise NotImplementedError''','''
import numpy as np
def attention(q,k,v):
    scores=q@k.T/np.sqrt(q.shape[-1]); mask=np.triu(np.ones(scores.shape,dtype=bool),1)
    scores=np.where(mask,-np.inf,scores); weights=np.exp(scores-scores.max(1,keepdims=True))
    weights/=weights.sum(1,keepdims=True)
    return weights@v,weights
''',[('uniform causal average',"o,w=attention(np.zeros((3,2)),np.zeros((3,2)),np.array([[2.],[4.],[9.]])); assert np.allclose(o[:,0],[2,3,5]); assert np.allclose(w.sum(1),1)")],[('future cannot leak',"q=np.eye(3); k=q.copy(); v=q.copy(); o,w=attention(q,k,v); v[-1]=100; oo,_=attention(q,k,v); assert np.allclose(o[:2],oo[:2]); assert np.allclose(np.triu(w,1),0)")],['Apply the triangular mask before the row softmax.','The first position can attend only to itself.'])
stage(gpt,'6 · Assemble and sample','Build a tiny causal transformer, fit a readout, and generate text.','Combine token and position embeddings, causal attention, residual connections, layer normalization, and a feed-forward block. Start with fixed random features and train the output projection: this is a deliberately small baseline, not full GPT pretraining. Fit your own short text, compare held-out loss, then sample with a seed. The optional PyTorch file trains all parameters with autograd on CPU or a CUDA device you explicitly choose. Write a model card: data origin, context length, loss, bias, and limits. Never include private notes in a public demo without reviewing them.', '''import numpy as np

class TinyTransformer:
    def __init__(self, size, width=8, context=16, seed=3):
        raise NotImplementedError
    def features(self, ids):
        raise NotImplementedError
    def logits(self, ids):
        raise NotImplementedError
    def fit(self, ids, steps=120, lr=0.3):
        raise NotImplementedError
    def generate(self, ids, count, seed=1):
        raise NotImplementedError''','''
import numpy as np
class TinyTransformer:
    def __init__(self,size,width=8,context=16,seed=3):
        r=np.random.default_rng(seed); self.context=context; self.size=size
        self.e=r.normal(0,.4,(size,width)); self.pos=r.normal(0,.1,(context,width))
        self.q=r.normal(0,.2,(width,width)); self.k=r.normal(0,.2,(width,width)); self.v=r.normal(0,.2,(width,width))
        self.f=r.normal(0,.2,(width,width)); self.out=np.zeros((width,size))
    def features(self,ids):
        ids=list(ids)[-self.context:]; x=self.e[ids]+self.pos[:len(ids)]
        z=(x@self.q)@(x@self.k).T/np.sqrt(x.shape[1]); z=np.where(np.triu(np.ones(z.shape),1),-np.inf,z)
        p=np.exp(z-z.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
        x=x+p@(x@self.v); x=(x-x.mean(1,keepdims=True))/np.sqrt(x.var(1,keepdims=True)+1e-5)
        return x+np.maximum(x@self.f,0)
    def logits(self,ids): return self.features(ids)@self.out
    def fit(self,ids,steps=120,lr=.3):
        if len(ids)<2: raise ValueError('Need two tokens')
        ids=list(ids)[:self.context+1]; h=self.features(ids[:-1]); y=np.array(ids[1:]); losses=[]
        for i in range(steps+1):
            z=h@self.out; p=np.exp(z-z.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
            losses.append(float(-np.log(p[np.arange(len(y)),y]).mean()))
            if i==steps: break
            p[np.arange(len(y)),y]-=1; self.out-=lr*h.T@p/len(y)
        return losses
    def generate(self,ids,count,seed=1):
        if not ids: raise ValueError('Supply a seed token')
        r=np.random.default_rng(seed); result=list(ids)
        for _ in range(count):
            z=self.logits(result)[-1]; p=np.exp(z-z.max()); p/=p.sum()
            result.append(int(r.choice(self.size,p=p)))
        return result
''',[('train and sample',"m=TinyTransformer(2); h=m.fit([0,1]*8); assert h[-1]<h[0]; a=m.generate([0],10); assert len(a)==11; assert set(a)<=set(range(2))")],[('causality and deterministic sampling',"m=TinyTransformer(3); assert np.allclose(m.features([0,1,2])[:2],m.features([0,1,0])[:2]); assert m.generate([0],8)==m.generate([0],8); assert m.logits([0,1]).shape==(2,3)")],['First verify causality with fixed weights before training.','Normalize over vocabulary columns; sample from the last position only.'])

shell=project('shell','Build your own shell',['Quoted parser','Commands and processes','Pipelines','Mini git: blobs, trees, commits, refs'],True,['Implemented a shell parser and process pipelines with explicit argument boundaries and exit-code checks.','Built a miniature content-addressed repository with immutable objects and branch references.'])
stage(shell,'1 · Words, quotes, pipes','Parse a command line into argument arrays.','A shell distinguishes quoted spaces from separators. Use shlex with pipe punctuation, then split only standalone pipe tokens. Reject empty commands: a trailing pipe has no right-hand process. Do not execute anything in this stage. Predict the result of echo "two words" | cat before running it.', '''import shlex

def parse(line):
    raise NotImplementedError''','''
import shlex
def parse(line):
    lexer=shlex.shlex(line,posix=True,punctuation_chars='|'); lexer.whitespace_split=True; lexer.commenters=''
    commands=[[]]
    for word in lexer:
        if word=='|':
            if not commands[-1]: raise ValueError('Empty command')
            commands.append([])
        else: commands[-1].append(word)
    if not commands[-1]:
        if len(commands)==1: return []
        raise ValueError('Trailing pipe')
    return commands
''',[('quoted arguments',"assert parse('echo \"two words\" | cat') == [['echo','two words'],['cat']]")],[('empty and malformed',"assert parse('  ')==[]\nfor line in ['| cat','echo a |']:\n    try: parse(line)\n    except ValueError: pass\n    else: raise AssertionError('empty command')")],['shlex handles quoting; you handle command boundaries.'])
stage(shell,'2 · Launch a process','Run an argument array and capture status and text.','A process receives an argument list, not a string to re-parse. subprocess.run with shell=False keeps quoted text intact and avoids an extra shell. Capture stdout, stderr and return code. Built-ins like cd would change your shell process; external commands cannot do that. Use sys.executable in tests so the same Python command works on Windows and macOS.', '''import subprocess

def execute(argv):
    # Return exit code, stdout, stderr. Timeout after 3 seconds.
    raise NotImplementedError''','''
import subprocess
def execute(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=3,shell=False)
    return p.returncode,p.stdout,p.stderr
''',[('stdout',"import sys\nc,o,e=execute([sys.executable,'-c','print(7)']); assert c==0 and o.strip()=='7'")],[('status and literal arguments',"import sys\nc,o,e=execute([sys.executable,'-c','import sys; print(sys.argv[1]); sys.exit(4)','a | b']); assert c==4 and o.strip()=='a | b'")],['Pass a list directly to subprocess.run.'])
stage(shell,'3 · Connect commands','Pass stdout through a deterministic pipeline.','A pipeline connects a producer to a consumer. This small cross-platform version buffers each completed command output, which is easier to debug than concurrent OS pipes. Return every exit code and the last stdout. This has a memory limit in real use: large streams need Popen pipes with careful draining and process cleanup. Explain that tradeoff in your write-up.', '''import subprocess

def pipeline(commands, text=''):
    # Return list of statuses and final stdout.
    raise NotImplementedError''','''
import subprocess
def pipeline(commands,text=''):
    codes=[]
    for argv in commands:
        p=subprocess.run(argv,input=text,capture_output=True,text=True,timeout=3,shell=False)
        text=p.stdout; codes.append(p.returncode)
    return codes,text
''',[('transform a stream',"import sys\nc,o=pipeline([[sys.executable,'-c','import sys; print(sys.stdin.read().upper(),end=\"\")']],'hello'); assert c==[0] and o=='HELLO'")],[('two stages and empty pipeline',"import sys\nc,o=pipeline([[sys.executable,'-c','print(6)'],[sys.executable,'-c','import sys; print(int(sys.stdin.read())*7)']]); assert c==[0,0] and o.strip()=='42'; assert pipeline([], 'x')==([], 'x')")],['Feed the previous stdout as the next input.'])
stage(shell,'4 · Keep a miniature history','Build content-addressed blobs, trees, commits and refs.','Git-like storage separates immutable objects from movable names. Hash a type, a NUL separator and payload bytes; store by digest. A tree maps names to blob IDs, a commit names a tree and parent, and a ref points to a commit. This teaching store uses SHA-256 and JSON, not the Git wire format. Keep serialization sorted so equal objects have equal IDs. Make a snapshot, change one file and explain which IDs change.', '''import hashlib, json
from pathlib import Path

class MiniGit:
    def __init__(self, root):
        raise NotImplementedError
    def put(self, kind, data):
        raise NotImplementedError
    def commit(self, files, parent=None):
        raise NotImplementedError
    def read(self, oid):
        raise NotImplementedError
    def set_ref(self, name, oid):
        raise NotImplementedError
    def get_ref(self, name):
        raise NotImplementedError''','''
import hashlib,json
from pathlib import Path
class MiniGit:
    def __init__(self,root): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def put(self,kind,data):
        raw=kind.encode()+b'\\0'+data; oid=hashlib.sha256(raw).hexdigest(); path=self.root/oid
        if not path.exists(): path.write_bytes(raw)
        return oid
    def commit(self,files,parent=None):
        tree={name:self.put('blob',data) for name,data in files.items()}
        tid=self.put('tree',json.dumps(tree,sort_keys=True).encode())
        return self.put('commit',json.dumps(dict(tree=tid,parent=parent),sort_keys=True).encode())
    def read(self,oid):
        if len(oid)!=64 or any(c not in '0123456789abcdef' for c in oid): raise ValueError('Bad object id')
        return (self.root/oid).read_bytes().split(b'\\0',1)
    def set_ref(self,name,oid):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name): raise ValueError('Bad ref')
        self.read(oid); (self.root/('ref-'+name)).write_text(oid)
    def get_ref(self,name):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name): raise ValueError('Bad ref')
        return (self.root/('ref-'+name)).read_text()
''',[('snapshot and ref',"import tempfile\nwith tempfile.TemporaryDirectory() as d:\n    g=MiniGit(d); c=g.commit({'a':b'hello'}); g.set_ref('main',c); assert g.get_ref('main')==c; assert g.read(c)[0]==b'commit'")],[('stable trees and parent',"import tempfile\nwith tempfile.TemporaryDirectory() as d:\n    g=MiniGit(d); a=g.commit({'a':b'x','b':b'y'}); assert a==g.commit({'b':b'y','a':b'x'}); b=g.commit({'a':b'z'},a); assert json.loads(g.read(b)[1])['parent']==a; assert g.read(a)[0]==b'commit'")],['Build blobs first, then a tree, then a commit.','A ref is a name you can move; an object is a value you keep.'])

agent=project('agent','Build your own AI agent',['Offline client contract','Tool dispatch with approval','Local memory','RAG over owner-selected notes','Grounded evals','CLI app'],False,['Draft after finishing: built an agent with explicit tool approval, local retrieval and grounded evaluations.'])
stage(agent,'1 · A client without spending','Build a request and a fake response adapter.','Before connecting any provider, define a client boundary you can test for free. A request carries a user message; the adapter returns text. Use a fake callable in this stage. Later stages will add explicit per-run approval, a spend cap, and an optional owner-configured local model. No credential discovery or network calls belong here. For RAG, retrieval will return passages with source locations rather than silently importing your notes.', '''def ask(message, transport):
    # Reject empty messages; call transport with a role/content dictionary.
    raise NotImplementedError''','''
def ask(message,transport):
    if not message.strip(): raise ValueError('Write a message')
    result=transport({'role':'user','content':message})
    if not isinstance(result,str): raise ValueError('Expected text')
    return result
''',[('fake adapter',"seen=[]\nassert ask('hello',lambda req: seen.append(req) or 'hi')=='hi'; assert seen==[{'role':'user','content':'hello'}]")],[('empty input never sends',"def nope(req): raise AssertionError('must not call')\ntry: ask('  ',nope)\nexcept ValueError: pass\nelse: raise AssertionError('empty message')")],['Inject a callable; do not import an API SDK yet.'])
drone=project('autopilot','Build a drone autopilot',['Fixed-step hover PID','Waypoint missions','Kalman sensor fusion','Wind and sensor faults','Optional owner-chosen hardware extension'],False,['Draft after finishing: implemented and evaluated deterministic hover and waypoint control in simulation.'])
stage(drone,'1 · Hold a height in simulation','Write a PID step with an explicit time interval.','Error is target minus measured height. Accumulate error times dt for the integral, and divide the change in error by dt for the derivative. Gains multiply these three signals. This stage returns a normalized correction, not a motor command. The Autopilot Lab can call this pure function at a fixed timestep. Real aircraft need saturation, anti-windup, failsafes and separate flight validation; this project starts entirely in simulation.', '''def pid(target, measured, integral, previous_error, dt, kp, ki, kd):
    # Return correction, updated integral, current error.
    raise NotImplementedError''','''
def pid(target,measured,integral,previous_error,dt,kp,ki,kd):
    if dt<=0: raise ValueError('Positive timestep required')
    error=target-measured; total=integral+error*dt
    return kp*error+ki*total+kd*(error-previous_error)/dt,total,error
''',[('proportional hover correction',"u,i,e=pid(2,1,0,1,.1,2,0,0); assert (u,i,e)==(2,.1,1)")],[('derivative integral and settling',"u,i,e=pid(1,1,2,1,.5,2,.5,.1); assert abs(u-.8)<1e-9 and i==2 and e==0\nheight=0.; velocity=0.; integral=0.; prev=1.\nfor _ in range(600):\n    u,integral,prev=pid(1,height,integral,prev,.01,4,0,3)\n    velocity+=u*.01; height+=velocity*.01\nassert abs(height-1)<.01")],['Compute the error before updating the integral.','Carry integral and previous_error between simulator frames.'])
game=project('game','Build a game and its opponent',['Lua fixed-step game core','LÖVE drawing and input','Python port','A* or behavior-tree opponent','RL experiment and demo'],False,['Draft after finishing: ported a Lua game to Python and compared a scripted opponent with a learned policy.'])
stage(game,'1 · Move on a game clock','Write a pure Lua movement step with arena bounds.','LÖVE calls update(dt) with elapsed seconds. Keep movement in a pure function so tests can run without a window. Position changes by velocity times dt, then clamps to the arena edges. Your later love.update calls this function and love.draw shows the result. In Python, dictionaries replace Lua tables and explicit return values stay the same. Keep collision and rendering separate.', '''function step(x, y, vx, vy, dt, width, height)
    error("Implement a movement step")
end''','''
function step(x,y,vx,vy,dt,width,height)
    assert(dt>=0 and width>=0 and height>=0)
    return math.max(0,math.min(width,x+vx*dt)), math.max(0,math.min(height,y+vy*dt))
end
''',[('move using seconds','local x,y=step(2,3,4,-2,0.5,10,10); assert(x==4 and y==2)')],[('clamp and zero time','local x,y=step(1,1,-20,30,1,8,8); assert(x==0 and y==8); x,y=step(2,3,4,4,0,8,8); assert(x==2 and y==3)')],['Multiply velocity by elapsed seconds, then clamp.'],language='lua')

contributions=project('contributions','Contributions: earn trust through evidence',['Read license and contribution guide','Reproduce and test','Fork and focused branch','Draft issue and PR etiquette','Respond to review','Morrowdock open core when approved'],False,['Replace with your own measured contribution results.'])
stage(contributions,'1 · Make a useful issue draft','Describe a reproducible observation before proposing a fix.','Read the license and CONTRIBUTING guide, search existing issues and reproduce locally before volunteering. Ask whether a small change fits the project. Keep one concern per patch, explain tests and tradeoffs, credit help and respond patiently to review. Never post generated code you cannot explain. Candidate queues: pytest contributor issues (https://github.com/pytest-dev/pytest/contribute) and NumPy contribution guide (https://numpy.org/devdocs/dev/index.html). Checked 2026-11-02; labels and availability may change, so inspect the issue before claiming it. For Morrowdock, use only the approved open-core repository and public fixtures after its license/public scope is confirmed; keep private learner data and paid credentials out. No public open-core availability is assumed.', '''def issue_draft(title, expected, observed, steps):
    raise NotImplementedError''', '''
def issue_draft(title,expected,observed,steps):
    if not title.strip() or not expected.strip() or not observed.strip() or not steps:
        raise ValueError('Add reproducible evidence')
    return {'title':title,'expected':expected,'observed':observed,'steps':list(steps)}
''',[('keep reproduction evidence',"d=issue_draft('Parser rejects a trailing pipe','clear message','empty launch',['parse a pipe']); assert d['steps']==['parse a pipe']")],[('incomplete drafts stay local',"try: issue_draft('x','expected','observed',[])\nexcept ValueError: pass\nelse: raise AssertionError('need steps')")],['Describe the observation and a small reproduction before the proposed patch.'])
gpt['supplementalFiles']={'MODEL-CARD.md':'# Tiny transformer model card\n\nData origin and permission:\nTraining / held-out split:\nContext and parameter count:\nDevice, seed, steps, training and held-out losses:\nSample prompt and output:\nBias, privacy, failure modes, and intended use:\n'}
gpt['optionalRevealFiles']={'torch_path.py':Path(__file__).with_name('torch_path.py').read_text(encoding='utf-8')}
from enhance_catalog import augment
augment(series)
from projects_quality_v2 import taught_series
series.append(taught_series())
from projects_gpt_v2 import taught_gpt_series
series.append(taught_gpt_series())
from projects_shell_v2 import taught_shell_series
series.append(taught_shell_series())
root=Path(__file__).resolve().parents[2]
raw=json.dumps(dict(schema=1,series=series),ensure_ascii=False,indent=2)+'\n'
(root/'tools/morrowdock-cli/morrowdock_cli/catalog.json').write_text(raw, encoding='utf-8', newline='\n')
native=root/'native-study/Core/Sources/StudyCore/Fixtures/projects-v1.json'
if native.parent.is_dir(): native.write_text(raw, encoding='utf-8', newline='\n')  # app fixture exists only in the private app repository
