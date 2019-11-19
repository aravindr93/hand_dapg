from hand_vil.local_settings import MAIN_DIR
import os
from hand_vil.config_main import DEFAULT_CONFIG


GEN_DATA_DIR = os.path.join(MAIN_DIR, 'gen_data')

RES_DIR = os.path.join(GEN_DATA_DIR, 'results')
VIDOES_FOLDER = os.path.join(RES_DIR, 'videos')
PLOTS_FOLDER = os.path.join(RES_DIR, 'plots')


DATA_DIR = os.path.join(GEN_DATA_DIR, 'data')
LOG_DIR = os.path.join(DATA_DIR, 'logs')

POLICIES_DIR = os.path.join(DATA_DIR, 'policies')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train_data')
VAL_DATA_DIR = os.path.join(DATA_DIR, 'val_data')

EXPERT_POLICIES_DIR = os.path.join(MAIN_DIR, '..', 'dapg', 'policies')
TRAIN_TRAJS = 5000
TEST_TRAJS = 100

ENV_ID = {
    'hand_pickup': 'mjrl_SHAP_slide_pickup-v42',
    'hand_hammer': 'mjrl_hammer-v0',
    'hand_pen': 'mjrl_pen_reposition-v2',
    'hand_door': 'mjrl_SHAP_door_handle-v5',
    'point_mass': 'mjrl_point_mass-v1',
}

EXPERT_POLICIES = {
    # Add  Other envs here
    'hand_pickup': 'relocate-v0.pickle',
    'hand_hammer': 'hammer-v0.pickle',
    'hand_pen': 'pen-v0.pickle',
    'hand_door': 'door-v0.pickle',
    # 'point_mass': 'point_mass.pickle',
}

ENTRY_POINT = {
    'hand_pickup': 'mj_envs.hand_manipulation_suite:RelocateEnvV0',
    'hand_hammer': 'mj_envs.hand_manipulation_suite:HammerEnvV0',
    'hand_pen': 'mj_envs.hand_manipulation_suite:PenEnvV0',
    'hand_door': 'mj_envs.hand_manipulation_suite:DoorEnvV0',
    # 'point_mass': 'mj_envs.hand_manipulation_suite:PointMassEnv',
}

VIZ_ENV_IDS = {
    'hand_hammer': 'hammer-v0',
    'hand_door': 'door-v0',
    'hand_pickup': 'relocate-v0',
    'hand_pen': 'pen-v0'
}

FRAME_SIZE = (128, 128)

CAMERA_NAME = {
    'hand_pickup': 'vil_camera',
    'hand_hammer': 'vil_camera',
    'hand_pen': 'vil_camera',
    'hand_door': 'vil_camera',
    'point_mass': 'top_view',
}
