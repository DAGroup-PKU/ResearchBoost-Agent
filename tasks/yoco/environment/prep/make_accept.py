"""Acceptance trees: copies of the reference with one known change each (python make_accept.py <ref tree> <dir>).

  setup_entry  (b) entry yoco/train.py sets TF32 before cli_main()            -> ERROR at gate 0 (setup guard)
  wd0          (c) weight decay set to 0 in the optimizer's param groups      -> INCORRECT (exact: Adam invariant)
  freeze_norm  (c) the final RMSNorm weight is excluded from the optimizer    -> INCORRECT (exact: missing tensors,
                   (skip the optimizer step for some parameters)                trainable-parameter digest)
  detach_half  (c) the gradient of the loss on the second half of every       -> statistical gate (forward identical)
                   sequence is dropped (logits detached there)
  rms_reorder  (d) RMSNorm x * rsqrt(m + eps) -> x / sqrt(m + eps)            -> not bitwise, statistical gate PASS
  lr102        (e) control arm: the optimizer applies 1.02 x the scheduled    -> exact checks hold, statistical gate
                   learning rate; the scheduler's value is restored after the     INCORRECT
                   step, so every exact item (scheduled LR, LR after the update) is untouched
"""
import shutil
import sys
from pathlib import Path

ref, base = Path(sys.argv[1]), Path(sys.argv[2])
EDITS = {
    'setup_entry': ('yoco/train.py',
                    'from fairseq_cli.train import cli_main\n',
                    'from fairseq_cli.train import cli_main\nimport torch\ntorch.backends.cuda.matmul.allow_tf32 = True\n'),
    'wd0': ('yoco/tasks/gpt.py',
            '        model.train()\n        model.set_num_updates(update_num)\n',
            '        model.train()\n        model.set_num_updates(update_num)\n'
            '        for group in optimizer.param_groups:   # acceptance (c): weight decay silently off\n'
            '            group["weight_decay"] = 0.0\n'),
    'freeze_norm': ('yoco/models/yoco.py',
                    '        model = YOCOModel(model)\n',
                    '        model = YOCOModel(model)\n'
                    '        model.model.layer_norm.weight.requires_grad_(False)   # acceptance (c): never updated\n'),
    'detach_half': ('yoco/models/decoder/yoco.py',
                    '        x = self.output_layer(x)\n\n        return x, None\n',
                    '        x = self.output_layer(x)\n'
                    '        h = x.size(1) // 2   # acceptance (c): no gradient from the second half of the loss\n'
                    '        x = torch.cat([x[:, :h], x[:, h:].detach()], dim=1)\n\n        return x, None\n'),
    'rms_reorder': ('yoco/models/decoder/rms_norm.py',
                    '        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n',
                    '        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n'),
    'lr102': ('yoco/tasks/gpt.py',
              '    def valid_step(self, sample, model, criterion):\n',
              '    def optimizer_step(self, optimizer, model, update_num):\n'
              '        # control arm: the optimizer applies 1.02 x the scheduled learning rate; the scheduler\'s value\n'
              '        # is restored right after the step (deliberately wrong training semantics)\n'
              '        lr = optimizer.get_lr()\n'
              '        optimizer.set_lr(lr * 1.02)\n'
              '        optimizer.step()\n'
              '        optimizer.set_lr(lr)\n\n'
              '    def valid_step(self, sample, model, criterion):\n'),
}
for name, (rel, old, new) in EDITS.items():
    if len(sys.argv) > 3 and name not in sys.argv[3:]:
        continue
    dest = base / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ref, dest, ignore=shutil.ignore_patterns('.git', '__pycache__', 'imgs'))
    f = dest / rel
    text = f.read_text()
    assert text.count(old) == 1, (name, text.count(old))
    f.write_text(text.replace(old, new))
    print('made', dest)
