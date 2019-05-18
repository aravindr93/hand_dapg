from hand_vil.run_utils import *
# Self-explanatory python stuff.
import time as timer
import json
import os
from hand_vil.settings import *


def main(config):
    dump(config)

    ts = timer.time()
    register_env(config)

    if config['train_expert']:
        train_expert_policy(config)
    print()
    dump(config)

    gen_data_from_expert(config)
    print()
    dump(config)

    do_dagger(config)
    print()
    dump(config)

    print('Done with all steps')
    print('total time taken = %f' % (timer.time() - ts))

def dump(config):
    config_file = os.path.join(config['main_dir'], 'config.json')

    with open(config_file, 'w') as fp:
        json.dump(config, fp)


if __name__ == '__main__':
    config = DEFAULT_CONFIG
    config['main_dir'] = os.path.join(DATA_DIR, '%s_%s' % (config['env_name'], config['id_post']))
    ensure_dir(config['main_dir'])
    config['id'] = '%s_id_%s' % (config['env_name'], config['id_post'])
    config['env_id'] = ENV_ID[config['env_name']]

    main(config)
