import pandas as pd
import subprocess, sys, os, time

start = time.time()
sub_dir=sys.argv[1]

cmd = 'pypy src/ensemble/ave2.py '+sub_dir
subprocess.call(cmd, shell=True) 

cmd = 'pypy src/ensemble/ave3.py '+sub_dir
subprocess.call(cmd, shell=True) 

cmd = 'pypy src/ensemble/ave5.py '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/ensemble/ave6.py '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/ensemble/ave7.py '+sub_dir
subprocess.call(cmd, shell=True)



cmd = 'pypy src/ensemble/ave10.py '+sub_dir
subprocess.call(cmd, shell=True)


cmd = 'pypy src/ensemble/ave13.py '+sub_dir
subprocess.call(cmd, shell=True)

