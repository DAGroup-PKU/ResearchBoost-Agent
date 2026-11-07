"""Check the added domain against the existing palette under CVD simulation.

Optional validation dependencies: numpy==1.26.4 colorspacious==1.1.2.
Run from the repo root: python docs/figures/check_3d_palette.py
Method: Machado CVD simulation at severity 100, clip display sRGB to [0,1],
then Euclidean CIELAB distance (Delta E 76). API follows:
https://colorspacious.readthedocs.io/en/latest/tutorial.html#simulating-colorblindness
"""
import ast
import json
from pathlib import Path

import numpy as np
from colorspacious import cspace_convert

module = ast.parse(Path(__file__).with_name('make_results_overview.py').read_text())
domains = next(ast.literal_eval(node.value) for node in module.body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == 'DOMAINS')
names = [name for name in domains if name != '3d'] + ['3d']
rgb = np.array([[int(domains[name][0][i:i+2], 16) / 255 for i in (1, 3, 5)] for name in names])
result = {'added_color': domains['3d'][0], 'metric': 'CIELAB Delta E 76',
          'cvd_model': 'Machado; severity 100; display sRGB clipped to [0,1]', 'conditions': {}}
for condition in ['normal', 'protanomaly', 'deuteranomaly', 'tritanomaly']:
    simulated = rgb if condition == 'normal' else np.clip(cspace_convert(
        rgb, {'name': 'sRGB1+CVD', 'cvd_type': condition, 'severity': 100}, 'sRGB1'), 0, 1)
    lab = cspace_convert(simulated, 'sRGB1', 'CIELab')
    existing = min(float(np.linalg.norm(lab[i]-lab[j])) for i in range(len(names)-1) for j in range(i+1, len(names)-1))
    added = min(float(np.linalg.norm(lab[-1]-other)) for other in lab[:-1])
    assert added >= existing, (condition, added, existing)
    result['conditions'][condition] = {'existing_minimum': existing, 'added_to_existing_minimum': added}
print(json.dumps(result, indent=2))
