import numpy as np
import matplotlib.pyplot as plt


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


def plot_success_episodes(path, interval=10):
    ep_rs_dqn = np.loadtxt("{}ep_rs_dqn.npy".format(path)).tolist()
    ep_rs_prior = np.loadtxt("{}ep_rs_prior.npy".format(path)).tolist()
    ep_success_dqn, ep_success_prior = [], []

    for ind_s in range(len(ep_rs_dqn) - interval):
        seg = ep_rs_dqn[ind_s:ind_s + interval]
        ep_success_dqn.append(max(sum(seg), 0)/interval)
    print(ep_success_dqn)
    for ind_s in range(len(ep_rs_prior) - interval):
        seg = ep_rs_prior[ind_s:ind_s + interval]
        ep_success_prior.append(max(sum(seg), 0)/interval)
    print(ep_success_prior)

    plt.plot(ep_success_dqn, label='dqn')
    plt.plot(ep_success_prior, label='dqn_prior')
    plt.title("success rate in episodes")
    plt.legend(loc='best')
    plt.show()


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
    print(states.reshape(10, 8, 8))
    print(actions)
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
    plot_success_episodes("../logs/map2/", interval=100)
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
