import pandas as pd
import subprocess, sys, os, time

start = time.time()
sub_dir=sys.argv[1]


cmd = 'pypy src/ensemble/ave41.py '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/ensemble/ave58.py '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/ensemble/ave91.py '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/ensemble/ave92.py '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/ensemble/ave93.py '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/ensemble/ave98.py '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/ensemble/ave99.py '+sub_dir
subprocess.call(cmd, shell=True)
