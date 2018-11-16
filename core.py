import env
import time
import random
import dqn as bt
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

my_env = env.TrafficEnv()

for i_episode in range(1000000):
    # listener()
    s = my_env.reset()

    s_sliding, s_others = s[0],np.array(s[1]+s[2])

    # fsm.Tick(command)
    k = 0
    ep_r = 0
    while True:
        action = bt.choose_action(s_sliding, s_others)
        # print("now_action",int(action))
        s, r, is_done, dist = my_env.step(action)
        s_sliding_, s_others_ = s[0], s[1]+s[2]
        # print(s_sliding_)

        bt.store_transition(s_sliding, s_others, action, r, s_sliding_, s_others_, is_done)

        k += 1
        ep_r += r

        if (bt.EPSILON < 0.9):
            bt.EPSILON += 0.000002
        if (bt.MEMORY_COUNTER > bt.MEMORY_CAPACITY):
            bt.learn()
            if is_done:
                print("Ep:", i_episode, "|Ep_r:", round(ep_r, 2), "|Epsilon", bt.EPSILON)
        if is_done:
            logger.info("collecting! ------Ep:", i_episode, "|Ep_r:", round(ep_r, 2), "|Epsilon", bt.EPSILON)
            print("collecting! ------Ep:", i_episode, "|Ep_r:", round(ep_r, 2), "|Epsilon", bt.EPSILON)
            break