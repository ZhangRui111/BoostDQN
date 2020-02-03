import numpy as np
from maze import Maze


def my_update(env):
    for t in range(10):
        s = env.reset()
        print(s)
        while True:
            env.render(0.1)
            a = np.random.random_integers(4)-1
            s, r, done, info = env.step(a)
            print('action:{0} | reward:{1} | done: {2}'.format(a, r, done))
            print(s)
            print('\n')
            if done:
                print(info)
                print("--------------------------------------")
                env.render(0.1)
                break

    # end of game
    print('game over')
    env.destroy()


def main():
    env = Maze('./maps/map1.json', full_observation=False)
    my_update(env)


if __name__ == '__main__':
    main()