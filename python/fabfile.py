from fabric.api import *

GITHUBURL='https://github.com/Ayutthaya/data-exploration-tools.git'

def synchronize_tools():
    run('rm -rf data-exploration-tools')
    run('mkdir -p data-exploration-tools')
    put('/home/nath/Projects/Kaggle/tools/*', '~/data-exploration-tools/')

def setconfigstring(configstring='default'):
    run('echo ' + str(configstring) + ' > ~/.configstring')

def prepare(data_path=None, configstring='default'):
    setconfigstring(configstring)
    synchronize_tools()
    with cd('data-exploration-tools/scripts'):
        run('chmod +x install-tools.sh')
        run('./install-tools.sh')
    if data_path is not None:
        upload(data_path)

def upload(data_path):
    print('data_path: %s' % data_path)
    put(data_path, '~/')
    run('tar -xzf data.tar.gz')
    run_feature_engineering()

def run_feature_engineering():
    synchronize_tools()
    run('mkdir -p results')
    run('cp data-exploration-tools/python/feature_engineering.py results/')
    run('~/anaconda3/bin/python data-exploration-tools/python/feature_engineering.py')
    run_train_eval()

def run_train_eval():
    synchronize_tools()
    run('mkdir -p results')
    run('cp data-exploration-tools/python/train_eval.py results/')
    run('~/anaconda3/bin/python -u data-exploration-tools/python/train_eval.py &> results/logs.txt')
    run_cv()

def run_cv():
    synchronize_tools()
    run('mkdir -p results')
    run('cp data-exploration-tools/python/cv.py results/')
    run('~/anaconda3/bin/python -u data-exploration-tools/python/cv.py &> results/logs.txt')
    compute_predictions()

def compute_predictions():
    synchronize_tools()
    run('mkdir -p results')
    run('~/anaconda3/bin/python -u data-exploration-tools/python/predictions.py')
    run('cp data-exploration-tools/python/*py results/')
    download_results()

def download_results():
    run('tar -czf results-$(date +%Y-%m-%d-%H-%M-%S)-$(hostname).tar.gz results')
    get('results*tar.gz')
