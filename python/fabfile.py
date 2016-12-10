from fabric.api import *

GITHUBURL='https://github.com/Ayutthaya/data-exploration-tools.git'

def prepare(data_path=None):
    run('git clone '+GITHUBURL)
    with cd('data-exploration-tools/scripts'):
        run('./install-tools.sh')
    if data_path is not None:
        upload(data_path)

def upload(data_path):
    print('data_path: %s' % data_path)
    put(data_path, '~/')
    run('tar -xzf data.tar.gz')
