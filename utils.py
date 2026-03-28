import os
import json
import math
import torch
import pandas as pd
import scipy.signal as signal

def convert_to_item(tensor):
    return tensor.cpu().detach().numpy().item()


def convert_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def convert_to_tensor(data, data_type, device):
    if device is None:
        return torch.as_tensor(data=data,
                               dtype=data_type)
    else:
        return torch.as_tensor(data=data,
                               dtype=data_type,
                               device=device)


def create_config_json(path, params):
    """
    create a txt file to record the config dictionary
    @param path: string
    @param params: dictionary, key/param name, value/param value
    """
    file = open(path, 'w')
    for k, v in params.items():
        file.write('{}: {}\n'.format(k, v))
    file.close()


def save_as_json(file, data):
    """
    Save data to the json file
    @ params: file: the json file that will be saved to
    @ params: data: data that need to be saved
    """
    data = json.dumps(data)
    with open(file, 'w+') as file:
        file.write(data)
    file.close()


def save_as_csv(file, data):
    """
    save the input data to a csv file
    @param file: file path
    @param data: input data
    @return:
    """
    data = pd.DataFrame(data)
    data.to_csv(file)


def load_json(file):
    """
    load saved data from json file
    @ params: file: the file that will be loaded
    """
    with open(file, 'r+') as file:
        content = file.read()

    return json.loads(content)


def check_dir(dir):
    """
    Check if the dir exists, otherwise create it
    @param dir:
    @return:
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_dirs(params):
    """
    Create all experiment directories
    @param params: class
    @return:
    """
    for k, v in params.__dict__.items():
        if 'PATH' in k.split('_') and len(k.split('_')) == 2:
            check_dir(dir=v)


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def calculate_distance(ego_coordinate, neigh_coordinate):
    return math.sqrt((ego_coordinate[0] - neigh_coordinate[0]) ** 2 + (ego_coordinate[1] - neigh_coordinate[1]) ** 2)


def set_env(env_name, start_seed):
    from metadrive import (
        MultiAgentMetaDrive,
        MultiAgentTollgateEnv,
        MultiAgentBottleneckEnv,
        MultiAgentIntersectionEnv,
        MultiAgentRoundaboutEnv,
        MultiAgentParkingLotEnv
    )

    envs_classes = dict(
        roundabout=MultiAgentRoundaboutEnv,
        intersection=MultiAgentIntersectionEnv,
        tollgate=MultiAgentTollgateEnv,
        bottleneck=MultiAgentBottleneckEnv,
        parkinglot=MultiAgentParkingLotEnv,
        pgma=MultiAgentMetaDrive
    )

    # For debugging
    # env = envs_classes[env_name.lower()](dict(num_agents=1, allow_respawn=False))
    # Use default environment settings
    # if env_name.lower() == 'bottleneck':
    #     env = envs_classes[env_name.lower()](dict(start_seed=start_seed,
    #                                               success_reward=10.0,
    #                                               out_of_road_penalty=5.0,
    #                                               crash_vehicle_penalty=5.0,
    #                                               crash_object_penalty=5.0,
    #                                               crash_sidewalk_penalty=0.0,
    #                                               driving_reward=1.0,  # default 1.0
    #                                               speed_reward=0.1,  # default 0.1
    #                                               crash_vehicle_cost=1,
    #                                               crash_object_cost=1,
    #                                               ))
    # else:
    #     env = envs_classes[env_name.lower()](dict(start_seed=start_seed))
    env = envs_classes[env_name.lower()](dict(start_seed=start_seed))

    return env


def set_eval_env(env_name, seed, num_agents):
    from metadrive import (
        MultiAgentMetaDrive,
        MultiAgentTollgateEnv,
        MultiAgentBottleneckEnv,
        MultiAgentIntersectionEnv,
        MultiAgentRoundaboutEnv,
        MultiAgentParkingLotEnv
    )

    envs_classes = dict(
        roundabout=MultiAgentRoundaboutEnv,
        intersection=MultiAgentIntersectionEnv,
        tollgate=MultiAgentTollgateEnv,
        bottleneck=MultiAgentBottleneckEnv,
        parkinglot=MultiAgentParkingLotEnv,
        pgma=MultiAgentMetaDrive,
    )

    # Use default environment settings
    env = envs_classes[env_name.lower()](dict(start_seed=seed, force_seed_spawn_manager=True, num_agents=num_agents))

    return env