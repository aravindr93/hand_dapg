import pickle
from mjrl.utils.gym_env import GymEnv
from settings import *
import mj_envs
from mj_envs.utils import hand_vil

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


CAMERA_NAME = 'vil_camera'
ENV_NAME = 'hand_door'
VIZ_FOLDER = 'hand_door_videos'
FULL_POLICY_PATH = ''


def main():
    e = GymEnv(VIZ_ENV_IDS[ENV_NAME])

    policy = pickle.load(open(FULL_POLICY_PATH, 'rb'))
    print('usind %d horizon', e.horizon)
    policy.model.eval()
    policy.old_model.eval()
    hand_vil.visualize_policy_offscreen(gym_env=e, save_loc=ensure_dir(os.path.join(VIDOES_FOLDER, VIZ_FOLDER)) + '/', policy=policy, use_tactile=False,
                                 num_episodes=3, horizon=e.horizon, mode='evaluation',
                                 camera_name=CAMERA_NAME, pickle_dump=False, frame_size_model=FRAME_SIZE)
    del (e)


if __name__ == '__main__':
    main()
