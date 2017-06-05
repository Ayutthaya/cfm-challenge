import os
from fabric.api import *


def set_configstring(configstring='default'):
    run('echo ' + str(configstring) + ' > ~/.configstring')

def get_path():
    return os.path.dirname(os.path.dirname(os.path.realpath(env['fabfile'])))

def upload(subfolder='python'):
    path = get_path()
    run('rm -rf data-exploration-tools/' + subfolder)
    run('mkdir -p data-exploration-tools/' + subfolder)
    put(path + '/' + subfolder + '/*', '~/data-exploration-tools/' + subfolder + '/')

def untar_data():
    with cd('data-exploration-tools/data'):
        run('tar -xzf data.tar.gz')
        run('mv data ~/data')

def prepare(data_path=None, configstring='default'):
    set_configstring(configstring)
    upload('python')
    upload('scripts')
    upload('data')
    untar_data()
    with cd('data-exploration-tools/scripts'):
        run('chmod +x install-tools.sh')
        run('./install-tools.sh')
    run_feature_engineering()

def run_feature_engineering():
    run('mkdir -p results')
    run('cp data-exploration-tools/python/feature_engineering.py results/')
    run('~/anaconda3/bin/python data-exploration-tools/python/feature_engineering.py')
    run_train_eval()

def run_train_eval():
    run('mkdir -p results')
    run('cp data-exploration-tools/python/train_eval.py results/')
    run('~/anaconda3/bin/python -u data-exploration-tools/python/train_eval.py &> results/logs.txt')
    run_cv()

def run_cv():
    run('mkdir -p results')
    run('cp data-exploration-tools/python/cv.py results/')
    run('~/anaconda3/bin/python -u data-exploration-tools/python/cv.py &> results/logs.txt')
    compute_predictions()

def compute_predictions():
    run('mkdir -p results')
    run('~/anaconda3/bin/python -u data-exploration-tools/python/predictions.py')
    run('cp data-exploration-tools/python/*py results/')
    download_results()

def download_results():
    run('tar -czf results-$(date +%Y-%m-%d-%H-%M-%S)-$(hostname).tar.gz results')
    get('results*tar.gz')
