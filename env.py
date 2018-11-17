import os,sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable SUMO_HOME")

config_path = "/home/jkwang/learn_sumo/quickstart/quickstart.sumo.cfg"
sumoBinary = "/usr/bin/sumo"
sumoguiBinary = "/usr/bin/sumo-gui"
sumoCmd = [sumoguiBinary,"-c",config_path,"--collision.action","remove","--start","--no-step-log","--no-warnings","--no-duration-log"]
import traci
import traci.constants as tc
import math
import numpy as np

class TrafficEnv(object):

    def __init__(self):
        traci.start(sumoCmd)

        self.cross_mapping={
            "-gneE1" : "cross_3",
            "gneE8" : "cross_4",
            "gneE10" : "cross_5",
            "gneE5" : "cross_2",
            "gneE6" : "cross_1"
        }
        self.light_mapping={
            "cross_3":-3,
            "cross_4":-3,
            "cross_5":-1,
            "cross_2":-1,
            "cross_1": 1
        }
        self.trafficPos_mapping={
            "cross_3": [-1000,0],
            "cross_4": [0,0],
            "cross_5": [1000,0],
            "cross_2": [1000,1000],
            "cross_1": [0,1000]
        }

        #Env --lanechange.duration
        self.step_num = 0
        self.AgentId = "agent"
        self.VehicleIds = []
        self.TotalReward = 0
        self.StartTime = 0
        self.end = 0
        #traci.vehicle.add("agent", "agent_route")
        traci.vehicle.setColor("agent", (255 , 0, 0, 255))
        traci.vehicle.setSpeed("agent",10)
        traci.gui.trackVehicle('View #0', "agent")

        #States
        self.Route = traci.vehicle.getRoute(self.AgentId)
        self.OccMapState = np.zeros((20, 7))
        self.VehicleState = [0,0,0]
        self.RoadState = [0 for i in range(9)]
        self.state = None

        #property to simulate
        self.end_x = 0
        self.end_y = 1000
        self.AgentX = 0
        self.AgentY = 0
        self.AgentSpeed = 10
        self.AgentAccRate = 2.0
        self.AgentDecRate = 1.5
        self.minLaneNumber = 0
        self.maxLaneNumber = 1
        self.oldDistance = 0
        self.nowDistance = 0

    def reset(self):
        self.end = 0
        self.TotalReward = 0
        self.oldDistance = 0
        self.nowDistance = 0

        traci.load(["-c",config_path,"--collision.action","remove","--no-step-log","--no-warnings","--no-duration-log"])
        print("Resetting...")
        #traci.vehicle.add("agent", "agent_route")
        traci.vehicle.setColor("agent", (255, 0, 0, 255))
        traci.vehicle.setSpeed("agent", 10)
        traci.gui.trackVehicle('View #0', "agent")

        traci.simulationStep()
        AgentAvailable = False
        while AgentAvailable == False:
            traci.simulationStep()
            self.VehicleIds = traci.vehicle.getIDList()
            if self.AgentId in self.VehicleIds:
                AgentAvailable = True
                self.StartTime = traci.simulation.getCurrentTime()
        for vehId in self.VehicleIds:
            traci.vehicle.subscribe(vehId,(tc.VAR_SPEED,tc.VAR_POSITION,tc.VAR_LANE_INDEX,tc.VAR_DISTANCE))
            traci.vehicle.subscribeLeader(self.AgentId,50)
            if vehId == self.AgentId:
                traci.vehicle.setSpeedMode(self.AgentId,0)
                traci.vehicle.setLaneChangeMode(self.AgentId,0)
        self.state = self.perception()

        return self.state

    def step(self,action):
        # define action:
        # action  |     meaning
        #    0    |    go straight
        #    1    |    break down
        #    2    |    change left
        #    3    |    change right
        #    4    |    do nothing

        position = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_POSITION]
        if (abs(position[0])+abs(position[1]))<999:
            self.maxLaneNumber = 2
        else:
            self.maxLaneNumber = 1

        self.end = 0
        reward = 0
        DistanceTravelled = 0
        if action == 0:
            maxSpeed = 16
            time = (maxSpeed - (traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED])) / self.AgentAccRate
            traci.vehicle.slowDown(self.AgentId, maxSpeed, time)
        elif action == 1:
            time = ((traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED]) - 0)/self.AgentDecRate
            traci.vehicle.slowDown(self.AgentId, 0, time)
        elif action == 2:
            laneindex = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_INDEX]
            if laneindex < self.maxLaneNumber:
                traci.vehicle.changeLane(self.AgentId,laneindex+1,100)
        elif action == 3:
            laneindex = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_INDEX]
            if laneindex > self.minLaneNumber:
                traci.vehicle.changeLane(self.AgentId,laneindex-1,100)
        elif action == 4:
            traci.vehicle.setSpeed(self.AgentId,traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED])
        traci.simulationStep()
        self.VehicleIds = traci.vehicle.getIDList()

        if self.AgentId in self.VehicleIds:
            for vehId in self.VehicleIds:
                traci.vehicle.subscribe(vehId,(tc.VAR_SPEED,tc.VAR_POSITION,tc.VAR_LANE_INDEX,tc.VAR_DISTANCE))
                traci.vehicle.subscribeLeader(self.AgentId,50)

            Vehicle_Params = traci.vehicle.getSubscriptionResults(self.AgentId)
            self.AutocarSpeed = Vehicle_Params[tc.VAR_SPEED]
            posAutox = Vehicle_Params[tc.VAR_POSITION]
            if math.sqrt((self.end_x-posAutox[0])**2+(self.end_y-posAutox[1])**2)<30:
                self.end = 100

            self.state = self.perception()
            reward = self.cal_reward(self.end)
        else:
            #self.state = self.perception()
            self.end = 1
            reward = self.cal_reward(is_collision=self.end)
            DistanceTravelled = 0


        return self.state, reward, self.end, DistanceTravelled

    def cal_reward(self,is_collision):
        if is_collision == 1:
            print("collision!")
            return -30
        elif is_collision == 100:
            print("arrive!")
            return 20
        else:
            self.nowDistance = traci.vehicle.getDistance(self.AgentId)
            del_distance = self.nowDistance - self.oldDistance
            reward = del_distance/500
            self.oldDistance = self.nowDistance

            return reward

    def perception(self):

        #the state is defined as:
        # 0   | 1    | 2    | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
        #speed|cos(a)|sin(a)|l? |r? |dis| r | y | g |l? | c? | r? |
        #

        self.VehicleIds = traci.vehicle.getIDList()

        AllVehicleParams = []

        #----------------------------to get the vehicle state------------------------
        for vehId in self.VehicleIds:
            traci.vehicle.subscribe(vehId, (tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_LANE_INDEX, tc.VAR_DISTANCE, tc.VAR_LANE_ID))
            VehicleParam = traci.vehicle.getSubscriptionResults(vehId)
            #AllVehicleParams.append(vehId)
            if vehId != self.AgentId:
                AllVehicleParams.append(VehicleParam)
            else:
                self.AgentSpeed = VehicleParam[tc.VAR_SPEED]
                self.AgentAngle = (VehicleParam[tc.VAR_ANGLE]/180)*math.pi
                self.AgentX = VehicleParam[tc.VAR_POSITION][0]
                self.AgentY = VehicleParam[tc.VAR_POSITION][1]
        self.VehicleState = [self.AgentSpeed,math.cos(self.AgentAngle),math.sin(self.AgentAngle)]

        #---------------------to calculate the occupanied state-----------------------
        LOW_X_BOUND = -6
        HIGH_X_BOUND = 6
        LOW_Y_BOUND = -10
        HIGH_Y_BOUND = 30
        self.OccMapState = np.zeros((20, 7))
        for VehicleParam in AllVehicleParams:
            VehiclePos = VehicleParam[tc.VAR_POSITION]
            rol = math.sqrt((VehiclePos[0]-self.AgentX)**2+(VehiclePos[1]-self.AgentY)**2)
            theta = math.atan2(VehiclePos[1]-self.AgentY,VehiclePos[0]-self.AgentX)
            reltheta = theta + self.AgentAngle
            relX = rol*math.cos(reltheta)
            relY = rol*math.sin(reltheta)
            if (relX>LOW_X_BOUND and relX<HIGH_X_BOUND) and (relY>LOW_Y_BOUND and relY<HIGH_Y_BOUND):
                indexX = int((6 + relX)/2 - 0.5)
                indexY = int((30 - relY)/2 - 0.5)
                self.OccMapState[indexY,indexX] = 1.0

        #-------------------------------to get the RoadState----------------------------
        #RoadState: [leftcan rightcan distance r y g leftava centerava rightava]
        self.RoadState = [1 for i in range(9)]
        now_laneindex = 0
        for vehId in self.VehicleIds:
            if vehId == self.AgentId:
                now_laneindex = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_INDEX]
                now_roadid = traci.vehicle.getRoadID(self.AgentId)
        if now_laneindex + 1 > self.maxLaneNumber:
            self.RoadState[0] = 0
        elif now_laneindex - 1 < self.minLaneNumber:
            self.RoadState[1] = 0

        try:
            nextTlsId = self.cross_mapping[now_roadid]
            rygState = traci.trafficlight.getRedYellowGreenState(nextTlsId)
            nextLight = rygState[self.light_mapping[nextTlsId]]
            x,y = self.trafficPos_mapping[nextTlsId][0],self.trafficPos_mapping[nextTlsId][1]
            x_v,y_v = traci.vehicle.getPosition(self.AgentId)
            distance = math.sqrt((x_v-x)**2+(y_v-y)**2)/1000
        except:
            nextLight='g'
            distance = 100

        self.RoadState[2] = distance
        if nextLight == 'g' or nextLight == 'G':
            self.RoadState[3] = 0
            self.RoadState[4] = 0
            self.RoadState[5] = 1
        elif nextLight == 'y' or nextLight == 'Y':
            self.RoadState[3] = 0
            self.RoadState[4] = 1
            self.RoadState[5] = 0
        else:
            self.RoadState[3] = 1
            self.RoadState[4] = 0
            self.RoadState[5] = 0

        for vehId in self.VehicleIds:
            if vehId == self.AgentId:
                nowLaneId = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_ID]
                links = traci.lane.getLinks(nowLaneId)
                for link in links:
                    okRoad = link[0][0:link[0].rfind("_")]
                    if okRoad in self.Route:
                        self.RoadState[7] = 1
                    else:
                        self.RoadState[7] = 0
                try:
                    leftLaneId = nowLaneId[0:-1] + str(int(nowLaneId[-1])+1)
                    links = traci.lane.getLinks(leftLaneId)
                    for link in links:
                        okRoad = link[0][0:link[0].rfind("_")]
                        if okRoad in self.Route:
                            self.RoadState[6] = 1
                        else:
                            self.RoadState[6] = 0
                except:
                    self.RoadState[6] = 0
                try:
                    rightLaneId = nowLaneId[0:-1] + str(int(nowLaneId[-1])-1)
                    links = traci.lane.getLinks(rightLaneId)
                    for link in links:
                        okRoad = link[0][0:link[0].rfind("_")]
                        if okRoad in self.Route:
                            self.RoadState[8] = 1
                        else:
                            self.RoadState[8] = 0
                except:
                    self.RoadState[8] = 0

        return [self.OccMapState,self.VehicleState,self.RoadState]

