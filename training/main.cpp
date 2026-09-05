#include <iostream>
#include <fstream>
#include <format>
#include <ranges>
#include <algorithm>
#include <random>
#include <thread>
#include <stop_token>
#include "neural_network.h"

template <typename ValueType>
struct std::formatter<std::vector<ValueType>>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const std::vector<ValueType>& vec, std::format_context& ctx) const
    {
        auto out = ctx.out();

        *out++ = '{';

        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            if (i != 0)
                out = std::format_to(out, ", ");

            out = std::format_to(out, "{}", vec[i]);
        }

        *out++ = '}';

        return out;
    }
};

int main()
{
    enum class CELL{START, END, OPEN, CLOSED};
    enum class ACTION{UP, DOWN, LEFT, RIGHT};

    constexpr int rows = 20,
                  columns = 30,
                  states = rows * columns,
                  actions = 4;

    CELL grid[rows][columns] = {
        { CELL::START, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED },
        { CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN },
        { CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::END }
    };

    neural_network mazeNN({2, 32, 32, 4}, 0.01);

    // thread of q-agent, which trains neural net:
    {
        std::jthread thr{[&](std::stop_token st) {
            double qValues[states][actions]{};
            constexpr double learningRate = 0.1,
                             discount = 0.95;
            double epsilon = 1;

            std::random_device rd;
            std::uniform_real_distribution<double> chanceDist(0, 1);
            std::uniform_int_distribution<int> actionDist(0, 3);

            while (!st.stop_requested())
            {
                int agentRow = 0,
                    agentColumn = 0,
                    stepsTaken = 0;

                while ((grid[agentRow][agentColumn] != CELL::END) && (stepsTaken < 150))
                {
                    const int state = agentRow * columns + agentColumn;
                    auto& stateQValues{qValues[state]};

                    int actionIndex;

                    if (chanceDist(rd) < epsilon)
                        actionIndex = actionDist(rd);
                    else
                        actionIndex = static_cast<int>(
                            std::ranges::max_element(stateQValues) - std::ranges::begin(stateQValues)
                        );

                    const ACTION actionTaken{static_cast<ACTION>(actionIndex)};

                    int newRow = agentRow,
                        newColumn = agentColumn;
                    double reward = -0.1;

                    switch (actionTaken)
                    {
                        case ACTION::UP:
                            if ((agentRow > 0) && (grid[agentRow - 1][agentColumn] != CELL::CLOSED))
                                newRow = agentRow - 1;
                            else
                                reward = -1;

                            break;
                        case ACTION::DOWN:
                            if ((agentRow + 1 < rows) && (grid[agentRow - 1][agentColumn] != CELL::CLOSED))
                                newRow = agentRow + 1;
                            else
                                reward = -1;

                            break;
                        case ACTION::LEFT:
                            if ((agentColumn > 0) && (grid[agentRow][agentColumn - 1] != CELL::CLOSED))
                                newColumn = agentColumn - 1;
                            else
                                reward = -1;

                            break;
                        case ACTION::RIGHT:
                            if ((agentColumn + 1 < columns) && (grid[agentRow][agentColumn + 1] != CELL::CLOSED))
                                newColumn = agentColumn + 1;
                            else
                                reward = -1;

                            break;
                    }

                    const int newState = newRow * columns + newColumn;
                    const auto& newStateQValues{qValues[newState]};

                    stateQValues[actionIndex]
                        += learningRate *
                            (reward + discount * std::ranges::max(newStateQValues) - stateQValues[actionIndex]);

                    math_vector<double> input(2, 0);
                    input[0] = agentRow / static_cast<double>(rows - 1);
                    input[1] = agentColumn / static_cast<double>(columns - 1);

                    math_vector<double> target(actions, 0);
                    target[actionIndex] = 1;

                    mazeNN.fit(input, target);

                    agentRow = newRow;
                    agentColumn = newColumn;
                    ++stepsTaken;
                }

                epsilon *= 0.995;
            }
        }};
        std::cin.get();
    }

    std::ofstream weightsFile{"/home/cartercpp/Documents/C++/NNLearnsQAgent/weights.txt"};
    weightsFile << std::format("{}", mazeNN.weights());

    std::ofstream biasesFile{"/home/cartercpp/Documents/C++/NNLearnsQAgent/biases.txt"};
    biasesFile << std::format("{}", mazeNN.biases());
}
