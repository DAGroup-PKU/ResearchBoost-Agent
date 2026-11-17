"""gunicorn configuration of the judge's GenEval reward service: upstream reward-server's gunicorn.conf.py (eight
synchronous workers, one on each GPU, timeout 120 s) with the address from R3_REWARD_BIND and the worker count from
R3_REWARD_GPUS (default 8). Both the reference and every candidate run against this same service."""
import os

NUM_DEVICES = int(os.environ.get('R3_REWARD_GPUS', '8'))
USED_DEVICES = set()
bind = os.environ.get('R3_REWARD_BIND', '127.0.0.1:18085')
workers = NUM_DEVICES
worker_class = 'sync'
timeout = 120


def pre_fork(server, worker):
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)


def post_fork(server, worker):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(worker.device_id)


def child_exit(server, worker):
    USED_DEVICES.discard(worker.device_id)
