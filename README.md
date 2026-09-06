# Neural Network Trained on a Q-Agent

A from-scratch C++ project where a **tabular Q-learning agent teaches a neural network** how to walk a maze.

The Q-table still learns the environment. After each Q-update, the network is trained to predict the Q-agent's current best action from a compact state encoding. No machine learning libraries are used.

This is part of the [ml-from-scratch](https://github.com/cartercpp/ml-from-scratch) collection.

## How It Works

The maze is a hardcoded `20x30` grid. A classic Q-table learns `Q[state][action]` with epsilon-greedy exploration.

In parallel, a network sees only a 6-dimensional encoding of the current cell:

```text
[normalized row, normalized column, can go up, can go down, can go left, can go right]
```

and is asked to output a one-hot target for the Q-table's current best action:

```text
6 inputs
    ↓
32 ReLU
    ↓
32 ReLU
    ↓
32 ReLU
    ↓
4 outputs  (UP, DOWN, LEFT, RIGHT)
```

So the Q-agent is the teacher. The network is the student.

```text
Q-table update
      ↓
best action for this cell
      ↓
one-hot target
      ↓
neural_network.fit(encoding, target)
```

This is different from a Deep Q-Network. The network is not estimating Q-values with a Bellman backup. It is imitating the policy that the Q-table has learned so far.

## Rewards and Exploration

The teacher uses tabular Q-learning:

```text
Q(s, a) += learningRate * (reward + discount * max Q(s', *) - Q(s, a))
```

| Event | Reward |
|---|---|
| Ordinary step | `-0.1` |
| Walk into a wall or off the map | `-1` (agent stays put) |
| Reach the goal | `+100` |

| Name | Default |
|---|---|
| `learningRate` | `0.1` |
| `discount` | `0.95` |
| starting `epsilon` | `1.0` |
| epsilon decay per episode | `* 0.999` |
| network learning rate | `0.005` |
| max steps per episode | `1000` |

## Training

The `training/` directory runs the teacher and student together.

1. The Q-agent picks an action (epsilon-greedy on the table).
2. It steps the maze and updates `Q(s, a)`.
3. The best action in `Q(s, *)` becomes a one-hot target.
4. The network is fit on the 6-feature encoding of `s`.
5. Press **Enter** to stop. The run writes `weights.txt`, `biases.txt`, and `qTable.txt`.

Network size:

```cpp
neural_network mazeNN({6, 32, 32, 32, 4}, 0.005);
```

## Demo

The `demo/` directory loads the trained network and walks the maze greedily in the terminal.

You will see:

- yellow `_` — start
- green `_` — goal
- red walls
- cyan `o` — the agent

The demo uses the network, not the Q-table. Press **Enter** to quit.

## Project Structure

```text
nn-trained-on-q-agent/
├── training/
│   ├── main.cpp
│   ├── neural_network.h
│   ├── neural_network.cpp
│   ├── matrix.h
│   ├── math_vector.h
│   ├── weights.txt
│   ├── biases.txt
│   └── qTable.txt
│
└── demo/
    ├── main.cpp
    ├── mazeNN.cpp
    ├── neural_network.h
    ├── neural_network.cpp
    ├── matrix.h
    ├── math_vector.h
    ├── weights.txt
    ├── biases.txt
    └── qTable.txt
```

## Implementation

* `matrix.h` / `math_vector.h` — the linear algebra used by backprop
* `neural_network` — multilayer perceptron with ReLU hidden layers
* `training/main.cpp` — maze, Q-table teacher, and supervised updates to the network

No TensorFlow, PyTorch, Eigen, or other ML libraries.

## Requirements

A modern C++ compiler with C++20/C++23 support (`std::jthread`, `<format>`, `<ranges>`).

A terminal that understands ANSI / truecolor helps for the demo.

## Why?

This sits between tabular Q-learning and a full DQN. The Q-table still does the reinforcement learning. The network learns a compact policy that can be replayed without looking up every cell in a table.
