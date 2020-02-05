import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from maze import Maze
from BoostDQN import BoostDQN
from DQN import DQN
from utils.utils import exist_or_create_folder, parse_model_config
from utils.data_analysis import plot_single_results, plot_results, plot_multi_results


def train(env, prior=False, save_ind=None, beta=None, prior_token=None):
    if prior:
        assert beta is not None, "Please provide beta!"
        assert prior_token is not None, "Please provide prior_token!"
        dqn = BoostDQN("./data/{}/".format(env.map_info), beta, prior_token)
    else:
        dqn = DQN("./data/{}/".format(env.map_info))

    # init_pos = [[4, 0], [0, 4], [9, 9]]  # map3
    init_pos = [[7, 0], [0, 6], [15, 13]]  # map4
    if beta is not None:
        writer = SummaryWriter(exist_or_create_folder("./logs/{}/{}_{}".format(env.map_info, dqn.info, beta)))
    else:
        writer = SummaryWriter(exist_or_create_folder("./logs/{}/{}".format(env.map_info, dqn.info)))

    ep_rs = []
    ep_r_steps = []
    step_rs = []
    steps_counter = 0

    params = parse_model_config("./data/{}/params.conf".format(env.map_info))
    max_ep = int(float(params["max_ep"]))
    max_steps = int(float(params["max_steps"]))

    writer.add_graph(dqn.eval_net, torch.rand(256).unsqueeze(0).to('cuda'))

    for i_episode in range(max_ep):
        if steps_counter > max_steps:
            break
        # Pick a start point randomly.
        s = env.reset([init_pos[np.random.randint(0, 3)]])
        ep_r = 0
        while True:
            env.render(0.001)
            a = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)
            dqn.store_transition(s, a, r, s_)
            ep_r += r
            s = s_

            steps_counter += 1
            writer.add_scalar("reward/step_r", r, steps_counter)
            step_rs.append(r)

            if dqn.memory_counter > (dqn.batch_size + 1):
                if dqn.info == 'dqn':
                    loss = dqn.learn()
                    writer.add_scalar("loss/loss", loss.item(), steps_counter)
                if dqn.info == 'prior':
                    loss, loss_dqn, loss_prior = dqn.learn()
                    writer.add_scalar("loss/loss", loss.item(), steps_counter)
                    writer.add_scalar("loss/loss_dqn", loss_dqn.item(), steps_counter)
                    writer.add_scalar("loss/loss_prior", loss_prior.item(), steps_counter)

            if done:
                dqn.update_epsilon()
                env.render(0.001)
                print("Ep: {} | Ep_r: {} | steps_counter: {}".format(i_episode, ep_r, steps_counter))
                writer.add_scalar("reward/ep_r", ep_r, steps_counter)
                writer.add_scalar("params/epsilon", dqn.epsilon, steps_counter)
                ep_rs.append(ep_r)
                ep_r_steps.append(steps_counter)
                break

    if save_ind is not None:
        if beta is None:
            np.savetxt(exist_or_create_folder("./logs/{}/{}/ep_rs_{}.npy"
                       .format(env.map_info, dqn.info, save_ind)), np.array(ep_rs))
            np.savetxt(exist_or_create_folder("./logs/{}/{}/ep_rs_step_{}.npy"
                       .format(env.map_info, dqn.info, save_ind)), np.array(ep_r_steps))
            np.savetxt(exist_or_create_folder("./logs/{}/{}/step_rs_{}.npy"
                       .format(env.map_info, dqn.info, save_ind)), np.array(step_rs))
        else:
            np.savetxt(exist_or_create_folder("./logs/{}/{}_{}/ep_rs_{}.npy"
                       .format(env.map_info, dqn.info, beta, save_ind)), np.array(ep_rs))
            np.savetxt(exist_or_create_folder("./logs/{}/{}_{}/ep_rs_step_{}.npy"
                       .format(env.map_info, dqn.info, beta, save_ind)), np.array(ep_r_steps))
            np.savetxt(exist_or_create_folder("./logs/{}/{}_{}/step_rs_{}.npy"
                       .format(env.map_info, dqn.info, beta, save_ind)), np.array(step_rs))
    else:
        if beta is None:
            np.savetxt(exist_or_create_folder("./logs/{}/{}/ep_rs.npy"
                       .format(env.map_info, dqn.info)), np.array(ep_rs))
            np.savetxt(exist_or_create_folder("./logs/{}/{}/ep_rs_step.npy"
                       .format(env.map_info, dqn.info)), np.array(ep_r_steps))
            np.savetxt(exist_or_create_folder("./logs/{}/{}/step_rs.npy"
                       .format(env.map_info, dqn.info)), np.array(step_rs))
        else:
            np.savetxt(exist_or_create_folder("./logs/{}/{}_{}/ep_rs.npy"
                       .format(env.map_info, dqn.info, beta)), np.array(ep_rs))
            np.savetxt(exist_or_create_folder("./logs/{}/{}_{}/ep_rs_step.npy"
                       .format(env.map_info, dqn.info, beta)), np.array(ep_r_steps))
            np.savetxt(exist_or_create_folder("./logs/{}/{}_{}/step_rs.npy"
                       .format(env.map_info, dqn.info, beta)), np.array(step_rs))


def main():
    global MAX_EP, MAX_STEPS
    env = Maze('./maps/map4.json', full_observation=True)
    # train(env, prior=False, save_ind=0)
    train(env, prior=True, save_ind=0, beta=0.6, prior_token=2)
    # for ind in [0, 1, 2, 3, 4]:
    #     train(env, prior=False, save_ind=ind)
    #     # train(env, prior=True, save_ind=ind, beta=0.1)
    # for beta in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    #     for ind in [0, 1, 2, 3, 4]:
    #         train(env, prior=True, save_ind=ind, beta=beta)


if __name__ == '__main__':
    main()
