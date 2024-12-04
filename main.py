import numpy as np
import random
import matplotlib.pyplot as plt


# State definitions
defaultState = ("Rested", "Undone", "8p")
physState = ["Rested", "Tired"]
hwState = ["Done", "Undone"]
timeState = ["10p", "8a", "10a"]
terminalState = "11a"

# Hard coded nodes
hardCodedNodes = [
    ("Tired", "Undone", "10p"),
    ("Rested", "Undone", "10p"),
    ("Rested", "Done", "10p"),
    ("Rested", "Undone", "8a"),
    ("Rested", "Done", "8a"),
    ("Tired", "Undone", "10a"),
    ("Rested", "Undone", "10a"),
    ("Rested", "Done", "10a"),
    ("Tired", "Done", "10a"),
]


# (Action, Reward, Transition Probability)
actions = {
    "Party": (+2, 1.0),
    "Rest": (0, 1.0),
    "Study": (-1, 1.0),
}

# (physState, hwState, timeState)
nodes = {
    (phys, hw, time): [
        action
        for action in actions
        if (phys == "Rested" and hw == "Undone")
        or (phys != "Rested" or hw != "Undone")
        and action != "Study"
    ]
    for time in timeState
    for hw in hwState
    for phys in physState
}

nodes = {key: value for key, value in nodes.items() if key in hardCodedNodes}


transitions = {
    (node, key) for node in nodes for key in actions.keys() if key in nodes[node]
}

# # Original approach to transitions
# transitions = {}
# for key in actions.keys():
#     for node in nodes:
#         r10 = actions[nodes[node][0]]
#         for i in range(len(nodes[node])):
#             if node[::2] == ("Rested", "10p") and nodes[node][i] == "Party":
#                 r10 = list(r10)
#                 r10[1] = 0.5
#                 r10 = tuple(r10)
#             transitions[(node, key)] = r10


# print(nodes)
print(transitions)
