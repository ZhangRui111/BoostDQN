import numpy as np
import matplotlib.pyplot as plt

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


def plot_results(path, interval=10, save_ind=0):
    # ep_rs_dqn = np.loadtxt("{}ep_rs_{}.npy".format(path, save_ind)).tolist()
    # ep_rs_prior = np.loadtxt("{}ep_rs_{}.npy".format(path, save_ind)).tolist()
    # ep_rs_dqn_steps = np.loadtxt("{}ep_rs_step_{}.npy".format(path, save_ind)).tolist()
    # ep_rs_prior_steps = np.loadtxt("{}ep_rs_step_{}.npy".format(path, save_ind)).tolist()
    # ep_success_dqn, ep_success_prior = [], []
    #
    # for ind_s in range(len(ep_rs_dqn) - interval):
    #     seg = ep_rs_dqn[ind_s:ind_s + interval]
    #     ep_success_dqn.append(max(sum(seg), 0)/interval)
    # # print(ep_success_dqn)
    # for ind_s in range(len(ep_rs_prior) - interval):
    #     seg = ep_rs_prior[ind_s:ind_s + interval]
    #     ep_success_prior.append(max(sum(seg), 0)/interval)
    # # print(ep_success_prior)
    #
    # plt.plot(ep_rs_dqn_steps[:len(ep_success_dqn)], ep_success_dqn, label='dqn')
    # plt.plot(ep_rs_prior_steps[:len(ep_success_prior)], ep_success_prior, label='dqn_prior')
    # plt.title("success rate in episodes")
    # plt.legend(loc='best')
    # plt.show()
    step_rs_dqn = np.loadtxt("{}dqn/ep_rs_step_r_{}.npy".format(path, save_ind)).tolist()
    step_rs_prior = np.loadtxt("{}prior/ep_rs_step_r_{}.npy".format(path, save_ind)).tolist()
    step_rs_dqn_ma, step_rs_prior_ma = [], []  # moving average

    for ind_s in range(interval - 1, len(step_rs_dqn)):
        seg = step_rs_dqn[ind_s-(interval-1):ind_s + 1]
        step_rs_dqn_ma.append(sum(seg)/interval)
    # print(step_rs_dqn_ma)
    for ind_s in range(interval - 1, len(step_rs_prior)):
        seg = step_rs_prior[ind_s - (interval - 1):ind_s + 1]
        step_rs_prior_ma.append(sum(seg) / interval)
    # print(step_rs_prior_ma)

    plt.plot(step_rs_dqn_ma, label='dqn')
    plt.plot(step_rs_prior_ma, label='dqn_prior')
    plt.title("Moving average of step_reward in {}".format(interval))
    plt.legend(loc='best')
    plt.show()


def plot_results_average(path, interval=10):
    dqn_ma_holder, prior_ma_holder = [], []
    for save_ind in [0, 1, 2, 3, 4]:
        step_rs_dqn = np.loadtxt("{}dqn/ep_rs_step_r_{}.npy".format(path, save_ind)).tolist()
        step_rs_prior = np.loadtxt("{}prior/ep_rs_step_r_{}.npy".format(path, save_ind)).tolist()
        step_rs_dqn_ma, step_rs_prior_ma = [], []  # moving average

        for ind_s in range(interval - 1, len(step_rs_dqn)):
            seg = step_rs_dqn[ind_s-(interval-1):ind_s + 1]
            step_rs_dqn_ma.append(sum(seg)/interval)
        # print(step_rs_dqn_ma)
        for ind_s in range(interval - 1, len(step_rs_prior)):
            seg = step_rs_prior[ind_s - (interval - 1):ind_s + 1]
            step_rs_prior_ma.append(sum(seg) / interval)
        # print(step_rs_prior_ma)

        dqn_ma_holder.append(step_rs_dqn_ma)
        prior_ma_holder.append(step_rs_prior_ma)

    dqn_ma_holder = list(map(np.array, dqn_ma_holder))
    prior_ma_holder = list(map(np.array, prior_ma_holder))
    # length trip
    dqn_ma_holder = [item[:39500] for item in dqn_ma_holder]
    prior_ma_holder = [item[:39500] for item in prior_ma_holder]
    # mean among several runs
    dqn_ma = np.array(dqn_ma_holder).mean(axis=0)
    prior_ma = np.array(prior_ma_holder).mean(axis=0)

    plt.plot(dqn_ma, label='dqn')
    plt.plot(prior_ma, label='dqn_prior')
    plt.title("Moving average of step_reward in {}".format(interval))
    plt.legend(loc='best')
    plt.show()


def write_priors(save_path, map_path, init_points, actions_target):
    env = Maze(map_path, full_observation=True)
    states_prior = []
    for point in init_points:
        s = env.reset([point])
        # env.render(0.01)
        # print(s.reshape(10, 10))
        states_prior.append(s)
    states = np.array(states_prior)
    actions = np.array(actions_target).reshape(states.shape[0], 1)
    # print(states.reshape(10, 10, 10))
    # print(actions)
    np.savetxt("{}states_prior.npy".format(save_path), states)
    np.savetxt("{}actions_target.npy".format(save_path), actions)


def naive_write_priors(path):
    states = [[0., 0., 0., 0., 0., 0., -1., 0.,
              0., 0., 0., 0., 0., 0., -1., 0.,
              0., -1., -1., -1., 0., 0., -1., 0.,
              0., 0., 1., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., -1., -1., -1., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 2., 0., 0.],

              [0., 0., 0., 0., 0., 0., -1., 0.,
              0., 0., 1., 0., 0., 0., -1., 0.,
              0., -1., -1., -1., 0., 0., -1., 0.,
              0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., -1., -1., -1., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 2., 0., 0.],

              [0., 0., 0., 0., 0., 0., -1., 1.,
               0., 0., 0., 0., 0., 0., -1., 0.,
               0., -1., -1., -1., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., -1., -1., -1., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 2., 0., 0.],

              [0., 0., 0., 0., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., -1., 0.,
               0., -1., -1., -1., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., 0., 1.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., -1., -1., -1., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 2., 0., 0.],

              [0., 0., 0., 0., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., -1., 0.,
               0., -1., -1., -1., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 1., 0., 0., 0., 0.,
               0., 0., 0., -1., -1., -1., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 2., 0., 0.],

              [0., 0., 0., 0., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., -1., 0.,
               0., -1., -1., -1., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., -1., -1., -1., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 2., 0., 1.],

              [0., 0., 0., 0., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., -1., 0.,
               0., -1., -1., -1., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., -1., -1., -1., 0., 0.,
               0., 0., 1., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 2., 0., 0.],

              [0., 0., 0., 0., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., -1., 0.,
               0., -1., -1., -1., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., -1., -1., -1., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 1., 0., 2., 0., 0.],

              [0., 0., 0., 0., 0., 0., -1., 0.,
               0., 0., 0., 1., 0., 0., -1., 0.,
               0., -1., -1., -1., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., -1., -1., -1., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 2., 0., 0.],

              [0., 0., 0., 0., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., -1., 0.,
               0., -1., -1., -1., 0., 0., -1., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., -1., -1., -1., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.,
               1., 0., 0., 0., 0., 2., 0., 0.]]

    actions = [1, 3, 1, 2, 3, 2, 1, 3, 3, 3]

    states = np.array(states)
    actions = np.array(actions).reshape(states.shape[0], 1)
    # print(states.reshape(10, 8, 8))
    # print(actions)
    np.savetxt("{}states_prior.npy".format(path), states)
    np.savetxt("{}actions_target.npy".format(path), actions)
    # states_reload = np.loadtxt("{}states_prior.npy".format(path))
    # actions_reload = np.loadtxt("{}actions_target.npy".format(path))
    # print("------------------------------------")
    # print(states_reload)
    # print(actions_reload)
    # print("------------------------------------")
    # print(states - states_reload)


def main():
    # naive_write_priors("../data/")
    init_points = [[2, 0], [2, 2], [2, 4], [2, 6], [2, 9],
                   [7, 0], [7, 2], [7, 4], [8, 6], [8, 8]]
    acts_t = [1, 2, 3, 1, 3, 1, 1, 1, 2, 0]
    write_priors("../data/map3/", "../maps/map3.json", init_points, acts_t)

    # plot_results("../logs/map2/", interval=100)

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
