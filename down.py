import logging, subprocess, time, os, re, signal, random


def shell(cmd, block=True, return_msg=True):
    import os
    my_env = os.environ.copy()
    home = os.path.expanduser('~')
    my_env['PATH'] = home + "/anaconda3/bin/:" + my_env['PATH']
    # print(my_env)
    logging.info('cmd is ' + cmd)
    if block:
        # subprocess.call(cmd.split())
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        if return_msg:
            msg = task.communicate()
            msg = [msg_.decode('utf-8') for msg_ in msg]
            if msg[0] != '':
                logging.info('stdout {}'.format(msg[0]))
            if msg[1] != '':
                logging.error('stderr {}'.format(msg[1]))
            return msg
        else:
            return task
    else:
        print('Non-block!')
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        return task


def my_wget(fid, fname):
    shell(f'rm -rf /tmp/cookies_{fid}.txt')
    task = shell(
        f"wget --quiet --save-cookies /tmp/cookies_{fid}.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={fid}' -O- ",
        return_msg=False
    )
    out, err = task.communicate()
    out = out.decode('utf-8')
    if len(re.findall(r'.*confirm=([0-9a-zA-Z_-]+).*', out)) == 0:
        print(out)
        print('no confirm continue')
        return 100
    confirm = re.findall(r'.*confirm=([0-9a-zA-Z_-]+).*', out)[0]
    print(confirm)
    if task.poll() != 0:
        raise ValueError('fail')
    task = shell(
        f"wget -c --load-cookies /tmp/cookies_{fid}.txt 'https://docs.google.com/uc?export=download&confirm={confirm}&id={fid}' -O {fname}",
        block=False)
    return task


if __name__ == '__main__':
    os.chdir('/data1/share/')
    fid = '0ByQS_kT8kViSZnZPY1dmaHJzMHc'
    fname = 'train_split.1.tar.gz.downloading'
    # fid = '1PvIM_FjDRVSKh_kRDiWBpKkNNp3fUH3o'
    # fname = 'kinetics_skeleton.zip'
    while True:
        try:
            time.sleep(random.randint(0, 3))
            task = my_wget(fid, fname)
            if task == 100:
                continue
            time.sleep(random.randint(19, 45))
            if task.poll() == 0:
                out, err = task.communicate(timeout=10)
                err = err.decode('utf-8')
                if 'The file is already fully retrieved; nothing to do' in err:
                    print('finish! ')
                    break
                else:
                    print(err, '!! task poll is ', task.poll())
                    continue
            os.killpg(os.getpgid(task.pid), signal.SIGTERM)
            out, err = task.communicate(timeout=10)
            err = err.decode('utf-8')
            print(err, 'task poll is ', task.poll())
        except Exception as e:
            print('exception: ', e, 'continue')
