import random


class TreeNode:
    def __init__(self, data, name, terminal=False):
        self.data = data
        # 0=P, 1=R, 2=S
        self.Party = []
        self.Rest = []
        self.Study = []
        self.name = name
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
num_states = 11  # Number of states
num_actions = 3  # Number of actions
V = {state: 0 for state in states}  # Initialize the value function

num_episodes = 50  # Number of episodes to run
alpha = 0.1  # Learning rate


def run_episode(printSteps=False):
    episode = []
    state = tree1  # Start from initial state
    while state != tree11:  # Continue until terminal state is reached
        # Choose action with equal probability
        action = random.choice(state.actions_possible())
        transitions = state.get_child(action)
        transition = random.choice(transitions)
        next_state, reward, probability = transition[0], transition[1], transition[2]
        episode.append((state, action, reward, probability))
        state = next_state
    if printSteps:
        print(f"\n{state}")
        for step, (state, action, reward, probability) in enumerate(episode, start=1):
            print(
                f"Episode Sequence {step}:\nState={state}, Action={action}, Reward={reward}, Probability={probability}"
            )
    return episode


# Run episodes
for episode_idx in range(num_episodes):
    episode = run_episode()
    total_reward = sum(r * p for (_, _, r, p) in episode)
    print(f"Episode {episode_idx+1} - Total reward: {total_reward}")

    # Update value function using first-visit Monte-Carlo
    visited = set()
    for i, (state, _, _, _) in enumerate(episode):
        if state not in visited:
            visited.add(state)
            reward = sum(r for (_, _, r, _) in episode[i:])
            V[state] = V[state] + alpha * (reward - V[state])

# Print value function
print("\nValue function:")
for i in V:
    print(f"{i}: {V[i]}")
# print(V)

# Calculate and print average reward for each episode
average_rewards = []
for episode_idx in range(num_episodes):
    episode = run_episode(True)
    average_reward = sum(r for (_, _, r, _) in episode) / len(episode)
    average_rewards.append(average_reward)

print("\nAverage reward for each episode:")
for i, avg in enumerate(average_rewards):
    print(f"Episode {i+1}: {avg}")
