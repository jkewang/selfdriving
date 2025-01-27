import env
import time
import random
import dqn_fc as bt
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
logger.addHandler(handler)

my_env = env.TrafficEnv()
bt.saver.restore(bt.sess,"./model/my-model.ckpt-3500")
f = open("./logger.txt",'w')
bt.EPSILON = 0.9999

for i_episode in range(1000000):
    # listener()
    s = my_env.reset()
    N_others = 12 * 10
    s_pre_others = np.zeros((N_others))
    s_pre_others2 = np.array(s[1] + s[2])
    for i in range(N_others):
        s_pre_others[i] = s_pre_others2[i % 12]

    s_sliding, s_others = s[0], s_pre_others

    # fsm.Tick(command)
    k = 0
    ep_r = 0

    while True:
        action = bt.choose_action(s_sliding, s_others)
        print("now_action",int(action))
        s, r, is_done, dist = my_env.step(action)

        print("reward=",r)
        s_pre_others2 = np.array(s[1] + s[2])
        for i in range(N_others):
            s_pre_others[i] = s_pre_others2[i % 12]

        s_sliding_, s_others_ = s[0], s_pre_others

        bt.store_transition(s_sliding, s_others, action, r, s_sliding_, s_others_, is_done)

        s_sliding = s_sliding_
        s_others = s_others_

        log = str(s_others)
        #print(log)

        k += 1
        ep_r += r

        if (bt.MEMORY_COUNTER > bt.MEMORY_CAPACITY):
            #bt.learn()
            if is_done:
                print("Ep:", i_episode, "|Ep_r:", round(ep_r, 2), "|Epsilon", bt.EPSILON)
        if is_done:
            print("collecting! ------Ep:", i_episode, "|Ep_r:", round(ep_r, 2), "|Epsilon", bt.EPSILON)
            break

f.close()