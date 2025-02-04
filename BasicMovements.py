from djitellopy import tello
from time import sleep
import KeyPressModule as kp

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

def getInput():
    if kp.getKey("t"): me.move_up(90)
    if kp.getKey("q"): me.land()
    if kp.getKey("e"): me.takeoff()

while True:
    getInput()
    me.send_rc_control(0,0,0,0)
    print(me.get_height(), me.get_speed_x(), me.get_speed_y(), me.get_speed_z())
