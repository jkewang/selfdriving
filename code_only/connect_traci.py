import os,sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable SUMO_HOME")

sumoBinary = "/usr/bin/sumo-gui"
sumoCmd = [sumoBinary,"-c","/home/jkwang/learn_sumo/quickstart/quickstart.sumo.cfg","--collision.action","remove"]
import traci
import traci.constants as tc

traci.start(sumoCmd)
step = 0
traci.vehicle.add("agent","!1")
traci.vehicle.setColor("agent",(255,0,0,255))
traci.vehicle.setSpeedMode("agent",0)
traci.vehicle.setSpeed("agent",10)
traci.vehicle.setLaneChangeMode("agent",0)
traci.gui.trackVehicle('View #0',"agent")
traci.vehicle.subscribe("agent", (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))
traci.vehicle.subscribeLeader("agent",50)
while step<1000:
    traci.simulationStep()
    VehicleIds = traci.vehicle.getIDList()
    print(VehicleIds)
    #if step == 50:
    #    traci.vehicle.changeLane("agent",1,10)
    #collision_vehicles = traci.simulation.getCollidingVehiclesNumber()
    #print(collision_vehicles)
    #if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
    traci.trafficlight.setRedYellowGreenState("cross_4","GrGrGrGrGrGrGrGrGrGrGrGr")
    print(traci.vehicle.getSubscriptionResults("agent"))
    #routeId = traci.vehicle.getRouteID("0")
    #print(routeId)
    step += 1
traci.close()

