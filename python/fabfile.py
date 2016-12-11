from fabric.api import *

GITHUBURL='https://github.com/Ayutthaya/data-exploration-tools.git'

def prepare(data_path=None, branch=None):
    run('git clone '+GITHUBURL)
    with cd('data-exploration-tools/scripts'):
        run('./install-tools.sh')
    if data_path is not None:
        upload(data_path, branch)

def upload(data_path, branch=None):
    print('data_path: %s' % data_path)
    put(data_path, '~/')
    run('tar -xzf data.tar.gz')
    if branch is not None:
        run_model(branch)

def run_model(branch):
    with cd('data-exploration-tools'):
        run('git pull origin master')
    run('mkdir -p results')
    run('cp data-exploration-tools/python/feature-engineering.py results/')
    run('~/anaconda3/bin/python data-exploration-tools/python/feature-engineering.py')
    run('cp data-exploration-tools/python/train-xgboost.py results/')
    run('~/anaconda3/bin/python data-exploration-tools/python/train-xgboost.py')
    run('tar -czf results-$(date +%Y-%m-%d-%H-%M-%S)-$(hostname).tar.gz results')
    get('results*tar.gz')
