//
// Created by cartercpp on 9/4/26.
//

#include "neural_network.h"
#include <stdexcept>
#include <vector>
#include <random>
#include <limits>
#include <utility>
#include <cstddef>
#include <cmath>
#include "math_vector.h"
#include "matrix.h"

math_vector<double> neural_network::Relu(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = (vec[i] > 0) ? vec[i] : 0;

    return vec;
}

math_vector<double> neural_network::ReluDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] > 0;

    return vec;
}

neural_network::neural_network(const std::vector<std::size_t>& neuronsPerLayer, double learningRate)
    : m_neuronsPerLayer{neuronsPerLayer}, m_learningRate{learningRate}
{
    const std::size_t layers = neuronsPerLayer.size();

    if (layers < 3)
        throw std::invalid_argument{"Must have at least an input layer, a hidden layer, and an output layer"};

    m_weightMatrices.reserve(layers);
    m_biasVectors.reserve(layers);

    std::random_device rd;

    for (std::size_t layer = 1; layer < layers; ++layer)
    {
        const std::size_t prevLayerSize = m_neuronsPerLayer[layer - 1],
                          layerSize = m_neuronsPerLayer[layer];

        matrix<double> weights(layerSize, prevLayerSize, 0);
        math_vector<double> biases(layerSize, 0);

        std::normal_distribution<double> dist(0, std::sqrt(1 / static_cast<double>(2 * prevLayerSize)));

        for (std::size_t i = 0; i < layerSize; ++i)
        {
            for (std::size_t i2 = 0; i2 < prevLayerSize; ++i2)
                weights[i][i2] = dist(rd);

            biases[i] = dist(rd);
        }

        m_weightMatrices.emplace_back(std::move(weights));
        m_biasVectors.emplace_back(std::move(biases));
    }
}

neural_network::neural_network(
    const std::vector<matrix<double>>& weights,
    const std::vector<math_vector<double>>& biases,
    const std::vector<std::size_t>& neuronsPerLayer,
    double learningRate
) : m_weightMatrices{weights}, m_biasVectors{biases}, m_neuronsPerLayer{neuronsPerLayer},
    m_learningRate{learningRate}
{}

std::vector<math_vector<double>> neural_network::Forward(const math_vector<double>& input) const
{
    const std::size_t layers = m_neuronsPerLayer.size();

    std::vector<math_vector<double>> activations;
    activations.reserve(layers);
    activations.push_back(input);

    for (std::size_t layer = 0; layer < layers - 1; ++layer)
    {
        if (layer + 1 < layers - 1)
            activations.emplace_back(Relu(m_weightMatrices[layer] * activations[layer] + m_biasVectors[layer]));
        else
            activations.emplace_back(m_weightMatrices[layer] * activations[layer] + m_biasVectors[layer]);
    }

    math_vector<double>& outputs{activations.back()};
    double max = std::numeric_limits<double>::lowest();

    for (std::size_t i = 0; i < outputs.size(); ++i)
        if (outputs[i] > max)
            max = outputs[i];

    double softmaxSum = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i)
    {
        outputs[i] = std::exp(outputs[i] - max);
        softmaxSum += outputs[i];
    }

    for (std::size_t i = 0; i < outputs.size(); ++i)
        outputs[i] /= softmaxSum;

    return activations;
}

math_vector<double> neural_network::predict(const math_vector<double>& input) const
{
    return Forward(input).back();
}

void neural_network::fit(const math_vector<double>& input, const math_vector<double>& target)
{
    const std::vector<math_vector<double>> activations{Forward(input)};
    const math_vector<double>& prediction{activations.back()};

    std::vector<matrix<double>> weightDeltas(m_weightMatrices.size());
    std::vector<math_vector<double>> biasDeltas(m_biasVectors.size());

    math_vector<double> activationDelta{prediction - target};
    for (std::size_t iter = 1; iter <= m_weightMatrices.size(); ++iter)
    {
        const std::size_t layer = m_weightMatrices.size() - iter;

        math_vector<double> temp{activationDelta};

        if (layer != m_weightMatrices.size() - 1)
            temp = temp.multiply(ReluDerivative(activations[layer + 1]));

        weightDeltas[layer] = outer_product(temp, activations[layer]);
        biasDeltas[layer] = temp;

        if (layer > 0)
            activationDelta = m_weightMatrices[layer].transpose() * temp;
    }

    for (std::size_t layer = 0; layer < m_weightMatrices.size(); ++layer)
    {
        m_weightMatrices[layer] -= m_learningRate * weightDeltas[layer];
        m_biasVectors[layer] -= m_learningRate * biasDeltas[layer];
    }
}

const std::vector<matrix<double>>& neural_network::weights() const
{
    return m_weightMatrices;
}

const std::vector<math_vector<double>>& neural_network::biases() const
{
    return m_biasVectors;
}