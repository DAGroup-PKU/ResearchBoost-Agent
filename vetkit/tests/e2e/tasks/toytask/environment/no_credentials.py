"""Toy credential check: the real tasks scan the container for keys and tokens; this one only looks at the environment."""
import os
import sys
bad = [k for k in os.environ if k.endswith(('_API_KEY', '_TOKEN'))]
sys.exit('credentials in the environment: %s' % bad if bad else 0)
