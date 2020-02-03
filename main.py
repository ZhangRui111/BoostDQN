import numpy as np
import matplotlib.pyplot as plt

from maze import Maze
from BoostDQN import BoostDQN
from DQN import DQN


MAX_EP = 2000


def train():
    dqn = BoostDQN()
    # dqn = DQN()

    init_pos = [[[3, 0]], [[0, 3]], [[7, 1]]]

    print('\nCollecting experience...')
    ep_rs = []
    for i_episode in range(MAX_EP):
        s = env.reset(init_pos[np.random.randint(0, 3)])
        ep_r = 0
        while True:
            env.render(0.001)
            a = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > (dqn.batch_size + 1):
                dqn.learn()

            if done:
                dqn.update_epsilon()
                env.render(0.001)
                print("Ep: {} | Ep_r: {} | Ep_epsilon: {}".format(i_episode, ep_r, dqn.epsilon))
                ep_rs.append(ep_r)
                break
            s = s_

    np.savetxt("./logs/ep_rs_prior.npy", np.array(ep_rs))
    # ep_success = []
    # for ind_s in range(len(ep_rs)-10):
    #     seg = ep_rs[ind_s:ind_s+10]
    #     ep_success.append(max(sum(seg), 0))
    # print(ep_success)
    # plt.plot(ep_success)
    # plt.title("success rate in episodes")
    # plt.show()


def main():
    global env
    env = Maze('./maps/map2.json', full_observation=True)
    env.after(100, train)  # Call function update() once after given time/ms.
    env.mainloop()  # mainloop() to run the application.


if __name__ == '__main__':
    main()
