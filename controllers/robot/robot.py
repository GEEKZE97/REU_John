# John Modl UROP Spring 2023: Towards Cooperative Intelligence in Multi-Agent Systems using Deep RL
# Adapted from the deepbots and deepworlds repositories: https://github.com/aidudezzz/deepbots
#                                                        https://github.com/aidudezzz/deepworlds
# Official citation:
#   @InProceedings{10.1007/978-3-030-49186-4_6,
#       author="Kirtas, M.
#       and Tsampazis, K.
#       and Passalis, N.
#       and Tefas, A.",
#       title="Deepbots: A Webots-Based Deep Reinforcement Learning Framework for Robotics",
#       booktitle="Artificial Intelligence Applications and Innovations",
#       year="2020",
#       publisher="Springer International Publishing",
#       address="Cham",
#       pages="64--75",
#       isbn="978-3-030-49186-4"
#   }
#----------------------------------------------------

from controller import Robot, Motor, DistanceSensor, Emitter, Receiver, Camera, Device, CameraRecognitionObject, Supervisor
import numpy as np
from deepbots.robots import CSVRobot


def normalize_to_range(value, min, max, new_min, new_max):
    value = float(value)
    min = float(min)
    max = float(max)
    new_min = float(new_min)
    new_max = float(new_max)
    return (new_max - new_min) / (max - min) * (value - max) + new_max


class FindTargetRobot(CSVRobot):
    def __init__(self, n_rangefinders):
        super(FindTargetRobot, self).__init__()
        self.setup_rangefinders(n_rangefinders)
        self.setup_motors()

        self.camera = Camera('camera1')
        self.camera.enable(32)          ### Enabled for basic timestep ###
        self.camera.recognitionEnable(32)

    def create_message(self):
        message = []
        for rangefinder in self.rangefinders:
            message.append(rangefinder.getValue())
        # print('robotRangerMessage',message)
        return message

    def use_message_data(self, message):
        self.motor_speeds[0] = float(message[0])*3
        self.motor_speeds[1] = float(message[1])*3
        self.motor_speeds = np.clip(self.motor_speeds, -6, 6)
        self._set_velocity(self.motor_speeds[0], self.motor_speeds[1])


        # # Action 1 is gas
        # gas = float(message[1])
        # # Action 0 is turning
        # wheel = float(message[0])
        #
        # # Mapping gas from [-1, 1] to [0, 4] to make robot always move forward
        # gas = (gas+1)*2
        # gas = np.clip(gas, 0, 4.0)
        #
        # # Mapping turning rate from [-1, 1] to [-2, 2]
        # wheel *= 2
        # wheel = np.clip(wheel, -2, 2)
        #
        # # Apply gas to both motor speeds, add turning rate to one, subtract from other
        # self.motor_speeds[0] = gas + wheel
        # self.motor_speeds[1] = gas - wheel
        #
        # # Clip final motor speeds to [-4, 4] to be sure that motors get valid values
        # self.motor_speeds = np.clip(self.motor_speeds, 0, 6)
        #
        # # Apply motor speeds
        # self._set_velocity(self.motor_speeds[0], self.motor_speeds[1])

    def setup_rangefinders(self, n_rangefinders):
        # Sensors
        self.n_rangefinders = n_rangefinders
        self.rangefinders = []
        self.ps_names = ['ps' + str(i) for i in range(self.n_rangefinders)
                        ]  # 'ps0', 'ps1',...,'ps7'

        for i in range(self.n_rangefinders):
            self.rangefinders.append(self.getDevice(self.ps_names[i]))
            self.rangefinders[i].enable(self.timestep)

    def setup_motors(self):
        # Motors
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self._set_velocity(0.0, 0.0)
        self.motor_speeds = [0.0, 0.0]
    
    def _set_velocity(self, v_left, v_right):
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)


robot_controller = FindTargetRobot(8)
robot_controller.run()
