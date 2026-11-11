"""Optional full-parameter transformer training. No API calls or downloads.
Install a PyTorch build suitable for your machine separately; CUDA is opt-in.
python torch_path.py my-reviewed-text.txt --device cpu
"""
import argparse
from pathlib import Path
import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, size, context=32, width=32):
        super().__init__()
        self.context=context
        self.token=nn.Embedding(size,width); self.position=nn.Embedding(context,width)
        self.block=nn.TransformerEncoderLayer(width,4,64,dropout=0,batch_first=True)
        self.norm=nn.LayerNorm(width); self.head=nn.Linear(width,size)
    def forward(self,x):
        n=x.shape[1]; h=self.token(x)+self.position(torch.arange(n,device=x.device))
        mask=torch.triu(torch.ones(n,n,device=x.device,dtype=torch.bool),diagonal=1)
        return self.head(self.norm(self.block(h,src_mask=mask)))

def main():
    p=argparse.ArgumentParser(); p.add_argument('text',type=Path); p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--steps',type=int,default=100)
    args=p.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable; choose cpu or install a compatible PyTorch build.')
    torch.manual_seed(7); text=args.text.read_text(encoding='utf-8'); vocab=sorted(set(text)); lookup={c:i for i,c in enumerate(vocab)}
    if len(text)<40: p.error('Use at least 40 characters of reviewed local text.')
    data=torch.tensor([lookup[c] for c in text],device=args.device); m=Transformer(len(vocab)).to(args.device); optimizer=torch.optim.AdamW(m.parameters(),lr=.003)
    for i in range(args.steps):
        start=int(torch.randint(len(data)-33,(1,)).item()); x=data[start:start+32][None]; y=data[start+1:start+33][None]
        loss=nn.functional.cross_entropy(m(x).reshape(-1,len(vocab)),y.reshape(-1)); optimizer.zero_grad(); loss.backward(); optimizer.step()
        if i%20==0: print(f'Step {i}: training loss {loss.item():.3f}')
    m.eval(); ids=data[:1][None]
    with torch.no_grad():
        for _ in range(120):
            prob=m(ids[:,-32:])[:,-1].softmax(-1); ids=torch.cat([ids,torch.multinomial(prob,1)],1)
    print(''.join(vocab[i] for i in ids[0].tolist()))
    # Write only model weights and vocabulary to this explicitly local artifact.
    torch.save(dict(weights=m.state_dict(),vocabulary=vocab), 'tiny-transformer.pt')
if __name__=='__main__': main()
