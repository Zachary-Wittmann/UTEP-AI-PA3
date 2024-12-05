class TreeNode:
    def __init__(self, data, name, terminal=False):
        self.data = data
        # 0=P, 1=R, 2=S
        self.Party = []
        self.Rest = []
        self.Study = []
        self.name = name
        self.value = 0
        self.terminal = terminal

    def get_child(self, action):
        if action == "P":
            return self.Party
        elif action == "R":
            return self.Rest
        elif action == "S":
            return self.Study

    def add_child(self, child, value, action, probability=1):
        if action == "P":
            self.Party.append((child, value, probability))
        elif action == "R":
            self.Rest.append((child, value, probability))
        elif action == "S":
            self.Study.append((child, value, probability))
        else:
            return False

    def actions_possible(self):
        actions = []
        if self.Party != []:
            actions.append("P")
        if self.Rest != []:
            actions.append("R")
        if self.Study != []:
            actions.append("S")
        return actions

    def __repr__(self):
        return f"TreeNode({self.data}, {self.name})"


# Make Tree nodes

# Layer 1
tree1 = TreeNode(20, "RU")

# Layer 2
tree2 = TreeNode(22, "TU")
tree3 = TreeNode(22, "RU")
tree4 = TreeNode(22, "RD")

# Layer 3
tree5 = TreeNode(8, "RU")
tree6 = TreeNode(8, "RD")

# Layer 4
tree7 = TreeNode(10, "TU")
tree8 = TreeNode(10, "RU")
tree9 = TreeNode(10, "RD")
tree10 = TreeNode(10, "TD")

# Layer 5 (Terminal State)
tree11 = TreeNode(11, "", True)

# Make transitions
tree1.add_child(tree2, 2, "P")
tree1.add_child(tree3, 0, "R")
tree1.add_child(tree4, -1, "S")

tree2.add_child(tree8, 2, "P")
tree2.add_child(tree5, 0, "R")

tree3.add_child(tree5, 0, "R")
tree3.add_child(tree5, 2, "P", 0.5)
tree3.add_child(tree8, 2, "P", 0.5)
tree3.add_child(tree6, -1, "S")

tree4.add_child(tree6, 0, "R")
tree4.add_child(tree6, 2, "P", 0.5)
tree4.add_child(tree9, 2, "P", 0.5)

tree5.add_child(tree7, 2, "P")
tree5.add_child(tree8, 0, "R")
tree5.add_child(tree9, -1, "S")

tree6.add_child(tree9, 0, "R")
tree6.add_child(tree10, 2, "P")

tree7.add_child(tree11, -1, "P")
tree7.add_child(tree11, -1, "R")
tree7.add_child(tree11, -1, "S")

tree8.add_child(tree11, 0, "P")
tree8.add_child(tree11, 0, "R")
tree8.add_child(tree11, 0, "S")

tree9.add_child(tree11, 4, "P")
tree9.add_child(tree11, 4, "R")
tree9.add_child(tree11, 4, "S")

tree10.add_child(tree11, 3, "P")
tree10.add_child(tree11, 3, "R")
tree10.add_child(tree11, 3, "S")

states = [tree1, tree2, tree3, tree4, tree5, tree6, tree7, tree8, tree9, tree10, tree11]

# Initialize values
values = {s: 0 for s in states}
actions_prefered = {s: None for s in states}
gamma = 0.99
threshold = 0.001
iteration = 0

# Value Iteration Loop
while True:
    max_change = 0
    for s in states:
        old_value = values[s]
        new_values = []
        actions = []
        for a in s.actions_possible():
            next_state = s.get_child(a)
            reward = 0
            for b in next_state:
                reward += b[2] * (b[1] + gamma * values[b[0]])
            new_values.append(reward)
            actions.append(a)
        if len(new_values) != 0:
            best_value = max(new_values)
            index = new_values.index(best_value)
            values[s] = best_value
            max_change = max(max_change, abs(best_value - old_value))
            actions_prefered[s] = actions[index]
            # # Additional detailed update information
            # print(f"Updating state {s}:")
            # print(f"Previous Value: {old_value}")
            # print(f"New Values: {new_values}")
            # print(f"Selected Action: {actions[index]}")
            # print(f"New Value: {best_value}")
            # print()

    iteration += 1
    if max_change < threshold:
        break


# Final output
print(f"Number of Iterations: {iteration}")
print("Final Values:")
for s in values:
    print(f"State {s}: Value = {values[s]}, Action = {actions_prefered[s]}")
