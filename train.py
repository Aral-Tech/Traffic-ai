from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# we need to import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("'SUMO_HOME' environment variable not defined, I'm unable to find tools, please declare environment variable manually. e.g export SUMO_HOME='/usr/share/sumo'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def get_vehicles_numbers(lanes):
    """
    Get the number of vehicles in each lane
    :param lanes: list of lanes (e.g. ["gneE0_0", "gneE0_1", "gneE0_2", "gneE0_3"])
    :return: dictionary of vehicles in each lane
    """
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
    return vehicle_per_lane


def get_waiting_time(lanes):
    """
    Get the waiting time of vehicles in each lane
    :param lanes: list of lanes (e.g. ["gneE0_0", "gneE0_1", "gneE0_2", "gneE0_3"])
    :return: total waiting time of vehicles in all lanes
    """
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


def phase_duration(junction, phase_time, phase_state):
    """
    Set the phase duration and phase state of the junction
    :param junction: junction id (e.g. "0")
    :param phase_time: duration of the phase (e.g. 6)
    :param phase_state: state of the phase (e.g. "GGGrrrrrrrrr")
    :return: None (this function does not return anything)
    """
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass of the model to get the Q values
        :param state: input state of the model (e.g. [1, 2, 3, 4])
        :return:
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions


class Agent:
    def __init__(
            self,
            gamma,
            epsilon,
            lr,
            input_dims,
            fc1_dims,
            fc2_dims,
            batch_size,
            n_actions,
            junctions,
            max_memory_size=100000,
            epsilon_dec=5e-4,
            epsilon_end=0.05,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = Model(
            self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions
        )
        self.memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "new_state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "reward_memory": np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=np.bool_),
                "mem_cntr": 0,
                "iter_cntr": 0,
            }

    def store_transition(self, state, state_, action, reward, done, junction):
        """
        Store the transition in the memory buffer of the agent for a specific junction number (e.g. 0)
        :param state: current state of the junction (e.g. [1, 2, 3, 4])
        :param state_: new state of the junction (e.g. [1, 2, 3, 4])
        :param action: action taken by the agent (e.g. 0)
        :param reward: reward received by the agent (e.g. -1)
        :param done: whether the episode is done or not
        :param junction: junction number (e.g. 0)
        :return:
        """
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        """
        Choose the action based on epsilon greedy policy
        :param observation: current state of the junction (e.g. [1, 2, 3, 4])
        :return: action taken by the agent
        """
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def reset(self, junction_numbers):
        """
        Reset the memory of the agent for a list of junction numbers
        :param junction_numbers: list of junction numbers (e.g. [0, 1, 2, 3])
        :return: None (this function does not return anything)
        """
        for junction_number in junction_numbers:
            self.memory[junction_number]['mem_cntr'] = 0

    def save(self, model_name):
        """
        Save the model to the disk
        :param model_name: name of the model (e.g. "model")
        :return: None (this function does not return anything)
        """
        torch.save(self.Q_eval.state_dict(), f'models/{model_name}.bin')

    def learn(self, junction):
        """
        Learn the Q values based on the memory of the agent for a specific junction number (e.g. 0)
        :param junction: junction number (e.g. 0)
        :return: None (this function does not return anything)
        """
        self.Q_eval.optimizer.zero_grad()

        batch = np.arange(self.memory[junction]['mem_cntr'], dtype=np.int32)

        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(
            self.Q_eval.device
        )
        new_state_batch = torch.tensor(
            self.memory[junction]["new_state_memory"][batch]
        ).to(self.Q_eval.device)
        reward_batch = torch.tensor(
            self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]["action_memory"][batch]

        q_eval = self.Q_eval.forward(state_batch)[batch, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = (
            self.epsilon - self.epsilon_dec
            if self.epsilon > self.epsilon_end
            else self.epsilon_end
        )


def run(train=True, model_name="model", epochs=50, steps=500):
    """
    Run the simulation for training or testing
    :param train: whether to train the model or not
    :param model_name: name of the model
    :param epochs: number of epochs
    :param steps: number of steps
    :return: None (this function does not return anything)
    """
    """execute the TraCI control loop"""
    epochs = epochs
    steps = steps
    best_time = np.inf
    total_time_list = list()
    traci.start(
        [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
    )
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))

    brain = Agent(
        gamma=0.99,
        epsilon=0.0,
        lr=0.1,
        input_dims=4,
        # input_dims = len(all_junctions) * 4,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions=4,
        junctions=junction_numbers,
    )

    if not train:
        brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin', map_location=brain.Q_eval.device))

    print(brain.Q_eval.device)
    traci.close()
    for e in range(epochs):
        if train:
            traci.start(
                [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )
        else:
            traci.start(
                [checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )

        print(f"epoch: {e}")
        select_lane = [
            ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
            ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
            ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
            ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
        ]

        # select_lane = [
        #     ["yyyyrrrrrrrrrrrr", "GGGGrrrrrrrrrrrr"],
        #     ["rrrryyyyrrrrrrrr", "rrrrGGGGrrrrrrrr"],
        #     ["rrrrrrrryyyyrrrr", "rrrrrrrrGGGGrrrr"],
        #     ["rrrrrrrrrrrryyyy", "rrrrrrrrrrrrGGGG"],
        # ]

        step = 0
        total_time = 0
        min_duration = 5

        traffic_lights_time = dict()
        prev_wait_time = dict()
        prev_vehicles_per_lane = dict()
        prev_action = dict()
        all_lanes = list()

        for junction_number, junction in enumerate(all_junctions):
            prev_wait_time[junction] = 0
            prev_action[junction_number] = 0
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction_number] = [0] * 4
            # prev_vehicles_per_lane[junction_number] = [0] * (len(all_junctions) * 4)
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        while step <= steps:
            traci.simulationStep()
            for junction_number, junction in enumerate(all_junctions):
                controled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controled_lanes)
                total_time += waiting_time
                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicles_numbers(controled_lanes)
                    # vehicles_per_lane = get_vehicle_numbers(all_lanes)

                    # storing previous state and current state
                    reward = -1 * waiting_time
                    state_ = list(vehicles_per_lane.values())
                    state = prev_vehicles_per_lane[junction_number]
                    prev_vehicles_per_lane[junction_number] = state_
                    brain.store_transition(state, state_, prev_action[junction_number], reward, (step == steps),
                                           junction_number)

                    # selecting new action based on current state
                    lane = brain.choose_action(state_)
                    prev_action[junction_number] = lane
                    phase_duration(junction, 6, select_lane[lane][0])
                    phase_duration(junction, min_duration + 10, select_lane[lane][1])

                    traffic_lights_time[junction] = min_duration + 10
                    if train:
                        brain.learn(junction_number)
                else:
                    traffic_lights_time[junction] -= 1
            step += 1
        print("total_time", total_time)
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save(model_name)

        traci.close()
        sys.stdout.flush()
        if not train:
            break
    if train:
        plt.plot(list(range(len(total_time_list))), total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total time")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()

def get_options():
    """
    Get the options from the command line
    :return: options from the command line (e.g. model_name, train, epochs, steps)
    """
    optParser = optparse.OptionParser(
        usage="%prog [options]",
        description="TrafficAI project from AralTech",
    )
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action='store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    run(train=train, model_name=model_name, epochs=epochs, steps=steps)
