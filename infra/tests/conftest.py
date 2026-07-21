import os
import sys

# infra/ is not an installed package; put it on the path so `import sync_rl_logs` works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
