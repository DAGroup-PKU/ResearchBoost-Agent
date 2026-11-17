"""Where the frozen setup ends and the per-update code begins, for the bench (gate 0) and the numerics adapter.

The real run is `<tree>/yoco/train.py` -> `fairseq_cli.train.cli_main()` -> `distributed_utils.call_main(cfg, main)`
-> `main(cfg)` (setup, then `while epoch_itr.next_epoch_idx <= max_epoch:` -> `train(cfg, trainer, task, epoch_itr)`)
-> `train()` (iterator / progress-bar setup, then `for i, samples in enumerate(progress):` = one update per pass).

The functions are taken LIVE, after importing the tree's modules exactly as the entry does (`import models`,
`import tasks`, `import criterions`, `from fairseq_cli.train import cli_main`), so a tree that vendors or
monkey-patches fairseq_cli is judged by what it actually runs. Frozen (must equal the reference AST):
  - the entry file `yoco/train.py` (whole module);
  - `cli_main` and `main` of the live `fairseq_cli.train` (whole functions);
  - `train` (unwrapped from its `@metrics.aggregate("train")` decorator) with the body of the per-update loop
    `for i, samples in enumerate(progress):` blanked; exactly one such loop.
Everything the loop body calls (Trainer.train_step, the task, criterion, model, optimizer, data loader) may change.
"""
import ast
import inspect
import runpy
import sys
import textwrap
from pathlib import Path

ENTRY = 'train.py'      # inside <tree>/yoco


def import_entry(script_dir):
    """Import what the entry imports, from the tree, without running its __main__ block. Returns the live
    fairseq_cli.train module."""
    script_dir = str(Path(script_dir).resolve())
    if sys.path[0] != script_dir:
        sys.path.insert(0, script_dir)
    runpy.run_path(str(Path(script_dir) / ENTRY), run_name='__yoco_entry__')
    mod = sys.modules.get('fairseq_cli.train')
    if mod is None:
        raise RuntimeError('the entry did not import fairseq_cli.train')
    return mod


def function_ast(fn):
    """(FunctionDef node, globals) of a live function, decorators unwrapped."""
    fn = inspect.unwrap(fn)
    src = textwrap.dedent(inspect.getsource(fn))
    node = ast.parse(src).body[0]
    if not isinstance(node, ast.FunctionDef):
        raise RuntimeError(f'{fn!r} is not a plain function')
    return node, fn.__globals__, inspect.getsourcefile(fn)


def is_update_loop(node):
    """`for i, samples in enumerate(progress):`"""
    return (isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == 'enumerate' and len(node.iter.args) == 1
            and isinstance(node.iter.args[0], ast.Name) and node.iter.args[0].id == 'progress'
            and isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2)


def split_train(train_node):
    """(statements before the loop, the loop node, statements after it) of train(); the loop must be top level."""
    for idx, stmt in enumerate(train_node.body):
        if is_update_loop(stmt):
            return train_node.body[:idx], stmt, train_node.body[idx + 1:]
    raise RuntimeError('train() has no top-level `for i, samples in enumerate(progress):` loop')


def split_main(main_node):
    """(statements before the epoch loop, the While node) of main()."""
    for idx, stmt in enumerate(main_node.body):
        if isinstance(stmt, ast.While):
            return main_node.body[:idx], stmt
    raise RuntimeError('main() has no top-level `while` epoch loop')


def dump(node):
    return ast.dump(node, include_attributes=False)


def fingerprint(script_dir):
    """The frozen-setup fingerprint of a tree (run in a fresh CPU process, see bench.py SETUP_PROBE)."""
    script_dir = Path(script_dir).resolve()
    out = {'entry': dump(ast.parse((script_dir / ENTRY).read_text()))}
    T = import_entry(script_dir)
    files = {}
    for name in ('cli_main', 'main', 'train'):
        node, _g, path = function_ast(getattr(T, name))
        files[name] = path
        if name == 'train':
            loops = sum(1 for n in ast.walk(node) if is_update_loop(n))
            out['train.update_loops'] = loops
            for n in ast.walk(node):
                if is_update_loop(n):
                    n.body = [ast.Pass()]
        out[name] = dump(node)
    return out, files
