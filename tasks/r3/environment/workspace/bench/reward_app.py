"""The GenEval reward service of the judge: upstream yifan123/reward-server (app_geneval.py @ 8245af2) unchanged in
what it computes, run from its checkout with the reward environment. Two additions, both outside the scoring:
the detector config and checkpoint paths come from the environment (R3_REWARD_DETECTOR_CONFIG,
R3_REWARD_DETECTOR_DIR; upstream edits two constants in reward_server/gen_eval.py by hand), and with
R3_REWARD_LOG set every request is appended to <R3_REWARD_LOG>.<pid>.jsonl: the SHA-256 of each JPEG image, of the
pickled metadata and only_strict flag, and of the pickled response (or the error). Started by bench.py with
bench/gunicorn_conf.py:  gunicorn -c <bench>/gunicorn_conf.py "reward_app:create_app()"  (cwd: the reward-server
checkout; PYTHONPATH: the bench directory and the checkout)."""
import hashlib
import json
import os
import pickle
import time
import traceback
from io import BytesIO

from flask import Blueprint, Flask, request
from PIL import Image

import reward_server.gen_eval as gen_eval

root = Blueprint('root', __name__)
INFERENCE_FN = None
LOG = None


def sha(b):
    return hashlib.sha256(b).hexdigest()


def create_app():
    global INFERENCE_FN, LOG
    gen_eval.MY_CONFIG_PATH = os.environ['R3_REWARD_DETECTOR_CONFIG']
    gen_eval.MY_CKPT_PATH = os.environ['R3_REWARD_DETECTOR_DIR']
    INFERENCE_FN = gen_eval.load_geneval()
    if os.environ.get('R3_REWARD_LOG'):
        LOG = open(f'{os.environ["R3_REWARD_LOG"]}.{os.getpid()}.jsonl', 'a', buffering=1)
    app = Flask(__name__)
    app.register_blueprint(root)
    return app


@root.route('/', methods=['POST'])
def inference():
    # identical to upstream app_geneval.inference, plus the log line
    data = request.get_data()
    entry = {'t': time.time(), 'request_sha256': sha(data)}
    try:
        data = pickle.loads(data)
        images = [Image.open(BytesIO(d), formats=['jpeg']) for d in data['images']]
        meta_datas = data['meta_datas']
        only_strict = data['only_strict']
        entry['images'] = [sha(d) for d in data['images']]
        entry['meta'] = sha(pickle.dumps((list(meta_datas), only_strict)))
        entry['prompts'] = [str(m.get('prompt', ''))[:200] if isinstance(m, dict) else '' for m in meta_datas]
        scores, rewards, strict_rewards, group_rewards, group_strict_rewards = INFERENCE_FN(images, meta_datas,
                                                                                            only_strict)
        response = {'scores': scores, 'rewards': rewards, 'strict_rewards': strict_rewards,
                    'group_rewards': group_rewards, 'group_strict_rewards': group_strict_rewards}
        entry['scores'] = [float(s) for s in scores]
        response = pickle.dumps(response)
        entry['response_sha256'] = sha(response)
        returncode = 200
    except Exception:
        response = traceback.format_exc()
        print(response)
        entry['error'] = response[-2000:]
        response = response.encode('utf-8')
        returncode = 500
    if LOG is not None:
        LOG.write(json.dumps(entry) + '\n')
    return response, returncode
