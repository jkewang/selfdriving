import env
import time
import random

myenv = env.TrafficEnv()
myenv.reset()

while(1):
    action = random.randint(0,4)
    s,r,end,dis = myenv.step(action)
    #time.sleep(1)
    if end == 1:
        myenv.reset()
        print("end!")
    elif end == 100:
        myenv.reset()
        print("arrive!")
