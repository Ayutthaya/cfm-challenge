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
    run('mkdir -p data')
    put(data_path, '~/data/')
    with cd('data'):
        run('for file in ./*.zip; do unzip $file; done')
