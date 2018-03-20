#!/usr/bin/python

import os, subprocess, signal

p = subprocess.Popen(['ps','-ax'], stdout = subprocess.PIPE)
out, err = p.communicate()

for line in out.splitlines():
    if 'StartAcquisition' in line:
        pid = int(line.split(None, 1)[0])
        os.kill(pid, signal.SIGKILL)
