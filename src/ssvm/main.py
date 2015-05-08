import logging, logging.handlers
import sys
from settings import *
from datetime import datetime
import subprocess, os
import re
import requests

def main():

    record = True
    print('Enter a name (default = kon) or konkon if you dont want to '
          'record')
    name = input()
    if name == 'konkon':
        record = False


    now = datetime.now()
    now_str = now.strftime('%m%d_%H%M%S')
    if name: now_str += '_' + name
    now_dir = os.path.join(OUTPUT_PATH, now_str)
    if record: os.mkdir(now_dir)
    rec_file = os.path.join(os.path.abspath(now_dir), 'w.out')


    SVM_FILE = 'svm_api'
    C = 100
    E = 100
    cmd = [SVM_LEARN_PATH, '--m', SVM_FILE, '-c', str(C), '-e', str(E),
           '_____yapsilon', rec_file]
    if not record: cmd = cmd[:-1]

    catchEps = re.compile(r'CEps=(\d+\.\d*),')
    catchData = re.compile(r'>>>\s*\[([^\]]*)\].*\(([^\)]*)\)')
    try:
        requests.post('http://140.112.18.227:5000/post/send_status', 
                     {'name': now_str, 
                      'status': 'new',
                      'type': 'svm',
                      'time_stamp': str(datetime.now())})
    except Exception:
        pass

    with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          bufsize=1) as p:
        try:
            for line in p.stdout:
                s = line.decode()
                print(s)
                mt = catchEps.search(s)
                if mt:
                    try:
                        requests.post('http://140.112.18.227:5000/post/send_data', 
                                     {'name': now_str, 
                                      'item': 'CEps',
                                      'value': mt.group(1),
                                      'time_stamp': str(datetime.now())})
                    except Exception:
                        pass
                mt = catchData.search(s)
                if mt:
                    try:
                        requests.post('http://140.112.18.227:5000/post/send_data', 
                                     {'name': now_str, 
                                      'item': mt.group(1),
                                      'value': mt.group(2),
                                      'time_stamp': str(datetime.now())})
                    except Exception:
                        pass

        except KeyboardInterrupt:
            try:
                requests.post('http://140.112.18.227:5000/post/send_status', 
                             {'name': now_str, 
                              'status': 'ended'})
            except Exception:
                pass
            p.terminate()
            os.kill(p.pid, 0)




if __name__ == '__main__':
    main()
