from djitellopy import tello
import KeyPressModule as kp
from time import sleep

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

def getKeyboardInput():
    lr,fb,ud,yw =0, 0, 0, 0
    speed=50
    if kp.getKey("LEFT"): lr=-speed
    elif kp.getKey("RIGHT"): lr=speed

    if kp.getKey("UP"): fb = -speed
    elif kp.getKey("DOWN"): fb=speed

    if kp.getKey("w"): ud = -speed
    elif kp.getKey("s"): ud = speed

    if kp.getKey("a"): yw = -speed
    elif kp.getKey("d"): yw = speed

    if kp.getKey("q"): me.land()
    if kp.getKey("e"): me.takeoff()

    return [lr,fb,ud,yw]


while True:
    vals=getKeyboardInput()
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    print(me.get_height(),me.get_speed_x(),me.get_speed_y(),me.get_speed_z())
    sleep(0.05)
