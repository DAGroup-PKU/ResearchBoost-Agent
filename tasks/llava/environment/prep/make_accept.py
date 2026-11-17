"""Acceptance trees: copies of the reference with one known change each (python make_accept.py <ref> <dir> [name ...]).

Every edit is anchored to text that occurs exactly once in haotian-liu/LLaVA@c121f04. Expected verdicts (bootstrap.py):

  setup_train          a line inside train()                               --quick        ERROR (setup_guard)
  sampler_no_accum     the length-grouped sampler forgets the accumulation factor, so other samples form each
                       optimizer step                                       --quick        INCORRECT (loader_check)
  projector0_detached  the first projector layer gets no gradient           --quick        INCORRECT (dead_gradient)
  rmsnorm_div_sqrt     RMSNorm x / sqrt(v + eps) instead of x * rsqrt(v + eps): equivalent, not bitwise
                                                                            --skip-timing  PASS (statistical gate)
  soft_labels          cross entropy with label smoothing 0.1 (the task forbids soft labels): every exact item holds,
                       only the statistical gate can reject it             --skip-timing  INCORRECT (statistical gate)

The last two patch transformers from a module of the tree that train_mem.py imports (the setup guard covers train.py
only), not site-packages.
"""
import shutil
import sys
from pathlib import Path

ENCODE = '''    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
'''
ENTRY = 'from llava.train.train import train\n'
ENTRY_PATCHED = 'from llava.train.acc_patch import apply\nfrom llava.train.train import train\n\napply()\n'

RMSNORM = '''"""Acceptance control: a mathematically equivalent RMSNorm."""
import torch
from transformers.models.llama import modeling_llama


def apply():
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    modeling_llama.LlamaRMSNorm.forward = forward
'''

SOFT_LABELS = '''"""Acceptance control: soft labels (label smoothing 0.1), which the task forbids."""
import functools

import torch
from transformers.models.llama import modeling_llama


def apply():
    modeling_llama.CrossEntropyLoss = functools.partial(torch.nn.CrossEntropyLoss, label_smoothing=0.1)
'''

EDITS = {
    'setup_train': [('llava/train/train.py',
                     'def train(attn_implementation=None):\n    global local_rank\n',
                     'def train(attn_implementation=None):\n    global local_rank\n'
                     '    torch.backends.cuda.matmul.allow_tf32 = True      # acceptance: a change inside train()\n')],
    'sampler_no_accum': [('llava/train/llava_trainer.py',
                          'world_size=self.args.world_size * self.args.gradient_accumulation_steps,',
                          'world_size=self.args.world_size,')],
    'projector0_detached': [('llava/model/llava_arch.py', ENCODE,
                             '''    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # acceptance: no gradient reaches the first projector layer (the second one still trains)
        p = self.get_model().mm_projector
        image_features = p[2](p[1](p[0](image_features).detach()))
        return image_features
''')],
    'rmsnorm_div_sqrt': [('llava/train/train_mem.py', ENTRY, ENTRY_PATCHED),
                         ('llava/train/acc_patch.py', None, RMSNORM)],
    'soft_labels': [('llava/train/train_mem.py', ENTRY, ENTRY_PATCHED),
                    ('llava/train/acc_patch.py', None, SOFT_LABELS)],
}


def main():
    ref, base = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    names = sys.argv[3:] or list(EDITS)
    for name in names:
        dest = base / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(ref, dest, ignore=shutil.ignore_patterns('.git', '__pycache__'))
        for rel, old, new in EDITS[name]:
            f = dest / rel
            if old is None:
                assert not f.exists(), (name, rel)
                f.write_text(new)
                continue
            text = f.read_text()
            assert text.count(old) == 1, (name, rel, text.count(old))
            f.write_text(text.replace(old, new))
        print('made', dest, flush=True)


if __name__ == '__main__':
    main()
