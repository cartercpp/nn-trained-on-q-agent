//
// Created by cartercpp on 9/4/26.
//

#ifndef NNLEARNSQAGENT_NEURAL_NETWORK_H
#define NNLEARNSQAGENT_NEURAL_NETWORK_H

#include <vector>
#include <cstddef>
#include "math_vector.h"
#include "matrix.h"

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(const std::vector<std::size_t>&, double);

    explicit neural_network(
        const std::vector<matrix<double>>&,
        const std::vector<math_vector<double>>&,
        const std::vector<std::size_t>&,
        double
    );

    // METHODS

    void fit(const math_vector<double>&, const math_vector<double>&);
    math_vector<double> predict(const math_vector<double>&) const;

    const std::vector<matrix<double>>& weights() const;
    const std::vector<math_vector<double>>& biases() const;

private:

    static math_vector<double> Relu(math_vector<double>);
    static math_vector<double> ReluDerivative(math_vector<double>);

    std::vector<math_vector<double>> Forward(const math_vector<double>&) const;

    std::vector<matrix<double>> m_weightMatrices;
    std::vector<math_vector<double>> m_biasVectors;
    std::vector<std::size_t> m_neuronsPerLayer;
    double m_learningRate;
};

#endif //NNLEARNSQAGENT_NEURAL_NETWORK_H