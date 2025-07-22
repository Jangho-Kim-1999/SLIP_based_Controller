import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
from numpy.linalg import inv
import os
import pandas as pd

xml_path = 'hopper.xml'
simend = 20

step_no = 0

FSM_AIR1 = 0
FSM_STANCE1 = 1
FSM_STANCE2 = 2
FSM_AIR2 = 3

fsm = FSM_AIR1

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# 전역 상태 변수
r_adm_hist = [0.0, 0.0]  # x[k-1], x[k-2]
f_hist = [0.0, 0.0, 0.0]  # F[k], F[k-1], F[k-2]

# 1. 루프 전에 로그 리스트 정의
log_time = []
log_r = []
log_force = []
log_ctrl = []

def admittance_controller(force_k):
    global r_adm_hist, f_hist

    # 시스템 파라미터
    M = 1.0     # 가상 질량
    B = 0    # 가상 감쇠
    K = 1000 #100.0   # 가상 스프링
    T = 0.001   # 시뮬레이션 timestep

    # 현재 외력 업데이트
    f_hist = [force_k] + f_hist[:2]  # shift and insert

    denom = 4*M + 2*B*T + K*T*T
    a1 = (8*M - 2*K*T*T) / denom
    a2 = (-4*M + 2*B*T - K*T*T) / denom
    b = (T*T) / denom

    # Difference equation
    x_k = (
        a1 * r_adm_hist[0] +
        a2 * r_adm_hist[1] +
        b * (f_hist[0] + 2*f_hist[1] + f_hist[2])
    )

    # 상태 업데이트
    r_adm_hist = [x_k] + r_adm_hist[:1]  # shift x[k-1], x[k-2]

    return x_k

def controller(model, data):
    global log_time, log_qpos, log_force, log_ctrl

    t = data.time
    fz = data.sensordata[0]

    rd = 0.75
    rd_adm = 0 #admittance_controller(fz)
    r_ref = rd - rd_adm

    K = 1
    r = 0.75 + data.qpos[2]
    u = K * (r_ref - r)
    data.ctrl[0] = u


    log_time.append(t)
    log_r.append(r)
    log_force.append(fz)
    log_ctrl.append(u)

def init_controller(model,data):
    """
    Optional: set initial control signal to 0
    """
    data.ctrl[0] = 0.0
    # data.qpos[]

def save_log_to_csv(log_time, log_r, log_force, log_ctrl, filename="log.csv"):
    df = pd.DataFrame({
        "time": log_time,
        "r": log_r,
        "force": log_force,
        "ctrl": log_ctrl
    })
    df.to_csv(filename, index=False)
    print(f"✅ {filename} 저장 완료")





def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 1.5])

init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    cam.lookat[0] = data.qpos[0] #camera will follow qpos
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

save_log_to_csv(log_time, log_r, log_force, log_ctrl)

glfw.terminate()
