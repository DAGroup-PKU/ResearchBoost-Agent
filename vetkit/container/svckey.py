"""svckey: the reference-pack key. One module, used on the host (bin/refpack.py) and
inside containers (svcpack.py); no dependencies beyond the standard library, Python >= 3.6.

    key = sha256(canonical JSON {schema: "rbb-pack/1", task, judge_hash, image_digest, gpu_model, world_size,
                                 driver, prep_version})

Canonical JSON: sort_keys, separators (",", ":"), ASCII. Field meanings (vetkit/README.md, "Reference packs"):
  judge_hash    sha256 over the judge workspace as the image ships it (/opt/researchboost/ws before preparation:
                every regular file except targets/ and __pycache__, "relpath\\0sha256\\n" in sorted order). This is
                prep_pack.workspace_hash() of llava/yoco applied to every task. It is NOT bench.py's evaluator_sha256,
                which also covers preparation output (records, task/*.json incl. ready.json) and so changes per run.
  image_digest  the registry manifest digest of the image ("sha256:<hex>") when it was pulled by digest, else
                "local:sha256:<image id>" (marked: not a registry digest).
  gpu_model     nvidia-smi name of the GPUs (all the same), e.g. "NVIDIA H200"
  world_size    bench.json world_size (the number of GPUs the task runs on)
  driver        nvidia-smi driver_version, e.g. "570.148.08"
  prep_version  "v<PREP_VERSION>" when bootstrap.py defines PREP_VERSION, else "bootstrap-sha256:<sha256 of the
                image's /opt/researchboost/bootstrap.py>"
"""
import hashlib
import json

SCHEMA = 'rbb-pack/1'
FIELDS = ('task', 'judge_hash', 'image_digest', 'gpu_model', 'world_size', 'driver', 'prep_version')


def key_fields(**fields):
    missing = [f for f in FIELDS if fields.get(f) in (None, '')]
    if missing:
        raise ValueError('pack key fields missing: %s' % missing)
    extra = sorted(set(fields) - set(FIELDS))
    if extra:
        raise ValueError('unknown pack key fields: %s' % extra)
    out = {f: fields[f] for f in FIELDS}
    out['world_size'] = int(out['world_size'])
    for f in FIELDS:
        if f != 'world_size':
            out[f] = str(out[f])
    out['schema'] = SCHEMA
    return out


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def pack_key(**fields):
    """The pack key (64 hex) of the given fields (svckey.FIELDS)."""
    return hashlib.sha256(canonical(key_fields(**fields)).encode()).hexdigest()


def image_digest_of(ref=None, image_id=None, repo_digest=None):
    """The image_digest field: the registry digest when the image was pulled by digest, else the local image ID marked
    as local."""
    if repo_digest:
        d = repo_digest.split('@', 1)[-1]
        if not d.startswith('sha256:'):
            raise ValueError('not a manifest digest: %r' % repo_digest)
        return d
    if ref and '@sha256:' in ref:
        return ref.split('@', 1)[1]
    if image_id:
        return 'local:' + (image_id if image_id.startswith('sha256:') else 'sha256:' + image_id)
    raise ValueError('no image digest or image id')


def sha256_file(path, block=8 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()
