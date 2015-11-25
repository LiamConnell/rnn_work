import subprocess
import os

def run_net(checkpoint , prime, testchar):
        aPath = '/home/ubuntu/kaggle/allen/char-rnn'
        os.chdir(aPath)
        cmd= ''.join(('th sample.lua ',checkpoint,' -primetext "', prime, '" -length 0 -testchar "',testchar,'" -gpuid -1'))
        output = float(subprocess.check_output(cmd, shell = True)[:-2])
        return output

