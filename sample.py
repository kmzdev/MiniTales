"""
MiniTales — geração de texto.
"""

import os
import pickle
import re
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

out_dir = 'out-tinystories-v2'
start = "once upon a time"
num_samples = 5
max_new_tokens = 200
temperature = 0.8
top_k = 50
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float16' if torch.cuda.is_available() else 'float32'
compile = False
dataset = 'tinystories'

config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())

torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

meta_path = os.path.join('data', dataset, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']


def encode(s):
    s = s.lower()
    words = re.findall(r"[a-z0-9]+|[.,!?;:'\"-]", s)
    return [stoi.get(w, 1) for w in words]


def decode(ids):
    return ' '.join(itos[i] for i in ids)


ckpt_path = os.path.join(out_dir, 'ckpt.pt')
ckpt = torch.load(ckpt_path, map_location=device)
model = GPT(GPTConfig(**ckpt['model_args']))
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('-' * 60)
