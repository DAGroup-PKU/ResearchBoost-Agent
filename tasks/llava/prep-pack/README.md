# Preparation packs

The task's Compose file mounts this directory read-only at `/opt/researchboost/prep-pack`. It ships empty: a pack
records the judge's frozen references on one GPU model and software stack, so a pack made elsewhere never matches
your machine, and the preparation then runs in full inside the container (see the task's doc for its duration).

After one full preparation on your hardware you can save it as a pack with `scripts/make_prep_pack.py` and put it
here; later trials on the same hardware and judge restore it instead of preparing again (a pack whose key does not
match is ignored, never trusted).
