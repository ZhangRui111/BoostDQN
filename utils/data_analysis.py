import numpy as np
import matplotlib.pyplot as plt


def get_moving_average(path, interval=500, trip=39000):
    """
    Get the moving average among 5 runs.
    :param path:
    :param interval:
    :param trip:
    :return:
    """
    ma_holder = []
    for save_ind in [0, 1, 2, 3]:
        step_rs = np.loadtxt("{}step_rs_{}.npy".format(path, save_ind)).tolist()
        step_rs_ma = []  # moving average
        for ind_s in range(interval - 1, len(step_rs)):
            seg = step_rs[ind_s-(interval-1):ind_s + 1]
            step_rs_ma.append(sum(seg)/interval)
        # print(step_rs_ma)
        ma_holder.append(step_rs_ma)

    ma_holder = list(map(np.array, ma_holder))
    ma_holder = [item[:int(trip)] for item in ma_holder]  # length trip
    ma = np.array(ma_holder).mean(axis=0)  # mean among several runs
    return ma


def plot_single_results(path, interval=500, beta=0.1, save_ind=0):
    """
    Plot the moving average of two settings by a single run.
    :param path:
    :param interval:
    :param save_ind:
    :return:
    """
    step_rs_dqn = np.loadtxt("{}dqn/step_rs_{}.npy".format(path, save_ind)).tolist()
    step_rs_prior = np.loadtxt("{}prior_{}/step_rs_{}.npy".format(path, beta, save_ind)).tolist()
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


def plot_results(path, interval=500, beta=0.1):
    """
    Plot the moving average of two settings by multiple runs.
    :param path:
    :param interval:
    :param beta:
    :return:
    """
    dqn_ma = get_moving_average("{}dqn/".format(path), interval)
    prior_ma = get_moving_average("{}prior_{}/".format(path, beta), interval)

    plt.plot(dqn_ma, label='dqn')
    plt.plot(prior_ma, label='dqn_prior')
    plt.title("Moving average of step_reward in {}".format(interval))
    plt.legend(loc='best')
    plt.show()


def plot_multi_results(path, interval=500):
    """
    Plot the moving average of all settings by multiple runs.
    :param path:
    :param interval:
    :return:
    """
    dqn_ma = get_moving_average("{}dqn/".format(path), interval)
    prior_0_1_ma = get_moving_average("{}token_1/".format(path), interval)
    prior_0_2_ma = get_moving_average("{}token_2/".format(path), interval)
    prior_0_4_ma = get_moving_average("{}token_3/".format(path), interval)
    # prior_0_6_ma = get_moving_average("{}prior_{}/".format(path, 0.6), interval)
    # prior_0_8_ma = get_moving_average("{}prior_{}/".format(path, 0.8), interval)
    # prior_1_0_ma = get_moving_average("{}prior_{}/".format(path, 1.0), interval)

    plt.plot(dqn_ma, label='dqn')
    plt.plot(prior_0_1_ma, label='10 states')
    plt.plot(prior_0_2_ma, label='18 states')
    plt.plot(prior_0_4_ma, label='a full trajectory')
    # plt.plot(prior_0_6_ma, label='delta-0.6')
    # plt.plot(prior_0_8_ma, label='delta-0.8')
    # plt.plot(prior_1_0_ma, label='delta-1.0')
    plt.title("Moving average of step_reward in {}".format(interval))
    plt.legend(loc='best')
    plt.xlabel('number of steps')  # plot figure's x axis name.
    plt.ylabel('moving average rewards')  # plot figure's y axis name.
    plt.show()


def main():
    # plot_single_results("../logs/map3/exp1/case1/", interval=500, beta=0.1, save_ind=0)
    # plot_results("../logs/map3/exp1/case1/", interval=500)
    plot_multi_results("../logs/map4/", interval=500)


if __name__ == '__main__':
    main()
