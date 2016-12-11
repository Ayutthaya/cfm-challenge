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
    run_model()

def run_model():
    with cd('data-exploration-tools'):
        run('git pull origin master')
    run('mkdir -p results')
    run('cp data-exploration-tools/python/feature-engineering.py results/')
    run('~/anaconda3/bin/python data-exploration-tools/python/feature-engineering.py')
    run('cp data-exploration-tools/python/train-xgboost.py results/')
    run('~/anaconda3/bin/python data-exploration-tools/python/train-xgboost.py')
    run('tar -czf results-$(date +%Y-%m-%d-%H-%M-%S)-$(hostname).tar.gz results')
    get('results*tar.gz')
