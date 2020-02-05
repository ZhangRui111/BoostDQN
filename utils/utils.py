import numpy as np
import os

from maze import Maze


def parse_model_config(path):
    """Parses the hyper-parameters configuration file"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    params_def = {}
    for line in lines:
        # if line.startswith('['):  # This marks the start of a new block
        #     module_defs.append({})
        #     module_defs[-1]['type'] = line[1:-1].rstrip()
        #     if module_defs[-1]['type'] == 'convolutional':
        #         module_defs[-1]['batch_normalize'] = 0
        # else:
        #     key, value = line.split("=")
        #     value = value.strip()
        #     module_defs[-1][key.rstrip()] = value.strip()
        key, value = line.split("=")
        value = value.strip()
        params_def[key.strip()] = value.strip()

    return params_def


def write_priors(save_path, map_path, prior_points, actions_target):
    """
    Write prior knowledge for a specific map.
    :param save_path:
    :param map_path:
    :param init_points:
    :param actions_target:
    :return:
    """
    env = Maze(map_path, full_observation=True)
    states_prior = []
    for ind in range(len(prior_points)):
        s = env.reset([prior_points[ind]])
        env.render(0.01)
        print(s.reshape(16, 16))
        print(actions_target[ind])
        states_prior.append(s)
    states = np.array(states_prior)
    actions = np.array(actions_target).reshape(states.shape[0], 1)
    # print(states.reshape(10, 10, 10))
    # print(actions)
    np.savetxt("{}states_prior.npy".format(save_path), states)
    np.savetxt("{}actions_target.npy".format(save_path), actions)


# def naive_write_priors(path):
#     states = [[0., 0., 0., 0., 0., 0., -1., 0.,
#               0., 0., 0., 0., 0., 0., -1., 0.,
#               0., -1., -1., -1., 0., 0., -1., 0.,
#               0., 0., 1., 0., 0., 0., 0., 0.,
#               0., 0., 0., 0., 0., 0., 0., 0.,
#               0., 0., 0., -1., -1., -1., 0., 0.,
#               0., 0., 0., 0., 0., 0., 0., 0.,
#               0., 0., 0., 0., 0., 2., 0., 0.],
#
#               [0., 0., 0., 0., 0., 0., -1., 0.,
#               0., 0., 1., 0., 0., 0., -1., 0.,
#               0., -1., -1., -1., 0., 0., -1., 0.,
#               0., 0., 0., 0., 0., 0., 0., 0.,
#               0., 0., 0., 0., 0., 0., 0., 0.,
#               0., 0., 0., -1., -1., -1., 0., 0.,
#               0., 0., 0., 0., 0., 0., 0., 0.,
#               0., 0., 0., 0., 0., 2., 0., 0.],
#
#               [0., 0., 0., 0., 0., 0., -1., 1.,
#                0., 0., 0., 0., 0., 0., -1., 0.,
#                0., -1., -1., -1., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., -1., -1., -1., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 2., 0., 0.],
#
#               [0., 0., 0., 0., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., -1., 0.,
#                0., -1., -1., -1., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., 0., 1.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., -1., -1., -1., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 2., 0., 0.],
#
#               [0., 0., 0., 0., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., -1., 0.,
#                0., -1., -1., -1., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 1., 0., 0., 0., 0.,
#                0., 0., 0., -1., -1., -1., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 2., 0., 0.],
#
#               [0., 0., 0., 0., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., -1., 0.,
#                0., -1., -1., -1., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., -1., -1., -1., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 2., 0., 1.],
#
#               [0., 0., 0., 0., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., -1., 0.,
#                0., -1., -1., -1., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., -1., -1., -1., 0., 0.,
#                0., 0., 1., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 2., 0., 0.],
#
#               [0., 0., 0., 0., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., -1., 0.,
#                0., -1., -1., -1., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., -1., -1., -1., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 1., 0., 2., 0., 0.],
#
#               [0., 0., 0., 0., 0., 0., -1., 0.,
#                0., 0., 0., 1., 0., 0., -1., 0.,
#                0., -1., -1., -1., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., -1., -1., -1., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 2., 0., 0.],
#
#               [0., 0., 0., 0., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., -1., 0.,
#                0., -1., -1., -1., 0., 0., -1., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                0., 0., 0., -1., -1., -1., 0., 0.,
#                0., 0., 0., 0., 0., 0., 0., 0.,
#                1., 0., 0., 0., 0., 2., 0., 0.]]
#
#     actions = [1, 3, 1, 2, 3, 2, 1, 3, 3, 3]
#
#     states = np.array(states)
#     actions = np.array(actions).reshape(states.shape[0], 1)
#     # print(states.reshape(10, 8, 8))
#     # print(actions)
#     np.savetxt("{}states_prior.npy".format(path), states)
#     np.savetxt("{}actions_target.npy".format(path), actions)
#     # states_reload = np.loadtxt("{}states_prior.npy".format(path))
#     # actions_reload = np.loadtxt("{}actions_target.npy".format(path))
#     # print("------------------------------------")
#     # print(states_reload)
#     # print(actions_reload)
#     # print("------------------------------------")
#     # print(states - states_reload)


def exist_or_create_folder(path_name):
    """
    Check whether a path exists, if not, then create this path.
    :param path_name: i.e., './logs/log.txt' or './logs/'
    :return: flag == False: failed; flag == True: successful.
    """
    pure_path = os.path.dirname(path_name)
    if not os.path.exists(pure_path):
        try:
            os.makedirs(pure_path)
        except OSError:
            pass
    return path_name


def main():
    # naive_write_priors("../data/")
    prior_points = [[7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [8, 4],
                    [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [8, 8], [7, 8], [6, 8]]
    acts_t = [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 2, 2, 2, 1]
    write_priors("../data/map4/3/", "../maps/map4.json", prior_points, acts_t)

    # for i in range(10):
    #     a = np.random.randint(0, 3)
    #     print(a)
    # a = [0, 1, 2, 3, 4, 3, 2, 1, 0]
    # import matplotlib.pyplot as plt
    # plt.plot(a)
    # plt.show()
    # print(a[1:3])


if __name__ == '__main__':
    main()
