from fabric.api import *

GITHUBURL='https://github.com/Ayutthaya/data-exploration-tools.git'

def synchronize_tools(branch):
    with cd('data-exploration-tools'):
        run('git fetch origin')
        run('git checkout origin/'+branch)

def setconfigstring(configstring='default'):
    run('echo ' + str(configstring) + ' > ~/.configstring')

def prepare(data_path=None, branch='master', configstring='default'):
    setconfigstring(configstring)
    run('git clone '+GITHUBURL)
    with cd('data-exploration-tools/scripts'):
        run('./install-tools.sh')
    if data_path is not None:
        upload(data_path, branch)

def upload(data_path, branch='master'):
    print('data_path: %s' % data_path)
    put(data_path, '~/')
    run('tar -xzf data.tar.gz')
    run_feature_engineering(branch)

def run_feature_engineering(branch='master'):
    synchronize_tools(branch)
    run('mkdir -p results')
    run('cp data-exploration-tools/python/feature_engineering.py results/')
    run('~/anaconda3/bin/python data-exploration-tools/python/feature_engineering.py')
    run_cv(branch)

def run_cv(branch='master'):
    synchronize_tools(branch)
    run('mkdir -p results')
    run('cp data-exploration-tools/python/cv.py results/')
    run('~/anaconda3/bin/python -u data-exploration-tools/python/cv.py &> results/logs.txt')
    compute_predictions(branch)

def compute_predictions(branch='master'):
    synchronize_tools(branch)
    run('mkdir -p results')
    run('~/anaconda3/bin/python -u data-exploration-tools/python/predictions.py')
    run('cp data-exploration-tools/python/*py results/')
    download_results()

def download_results():
    run('tar -czf results-$(date +%Y-%m-%d-%H-%M-%S)-$(hostname).tar.gz results')
    get('results*tar.gz')
