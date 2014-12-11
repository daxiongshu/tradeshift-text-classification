import subprocess, sys, os, time

start = time.time()
data_dir=sys.argv[1]
sub_dir=sys.argv[2]

# private LB 0.0057688 this is the best single online model
cmd = 'pypy src/online-model/tk36x_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True) 

# not submitted by itself
cmd = 'pypy src/online-model/tk33_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True) 

# not submitted by itself
cmd = 'pypy src/online-model/tk16_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True) 

# not submitted by itself
cmd = 'pypy src/online-model/tk19_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True) 


cmd = 'pypy src/online-model/tk20_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True) 

cmd = 'pypy src/online-model/nn2_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True) 

cmd = 'pypy src/online-model/tk5_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/online-model/tk6_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/online-model/tk7_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/online-model/tk8_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/online-model/tk9_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True)


cmd = 'pypy src/online-model/tk28_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True)

cmd = 'pypy src/online-model/nmy_solution.py '+data_dir+' '+sub_dir
subprocess.call(cmd, shell=True)





