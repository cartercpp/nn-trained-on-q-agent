//
// Created by cartercpp on 9/4/26.
//

#ifndef NNLEARNSQAGENT_MATH_VECTOR_H
#define NNLEARNSQAGENT_MATH_VECTOR_H

#include <format>
#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <concepts>
#include <cstddef>

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class math_vector
{
public:

    // CONSTRUCTORS

    math_vector() = default;

    explicit math_vector(std::size_t size, ValueType value = 0)
        : m_data(size, value)
    {}

    math_vector(const std::initializer_list<ValueType> data)
        : m_data{data}
    {}

    // METHODS

    [[nodiscard]] std::size_t size() const noexcept
    {
        return m_data.size();
    }

    ValueType& operator[](std::size_t index)
    {
        if (index >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[index];
    }

    const ValueType& operator[](std::size_t index) const
    {
        if (index >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[index];
    }

    math_vector multiply(const math_vector& other) const
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"Cannot multiply 2 vectors with different sizes"};

        math_vector output(m_data.size(), 0);
        for (std::size_t i = 0; i < m_data.size(); ++i)
            output[i] = m_data[i] * other[i];

        return output;
    }

    math_vector& operator*=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef *= scalar;});
        return *this;
    }

    math_vector& operator/=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef /= scalar;});
        return *this;
    }

    math_vector& operator+=(const math_vector& other)
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"Cannot add 2 vectors with different sizes"};

        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] += other[i];

        return *this;
    }

    math_vector& operator-=(const math_vector& other)
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"Cannot add 2 vectors with different sizes"};

        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] -= other[i];

        return *this;
    }

private:

    std::vector<ValueType> m_data;
};

template <typename ValueType>
struct std::formatter<math_vector<ValueType>>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const math_vector<ValueType>& vec, std::format_context& ctx) const
    {
        auto out = ctx.out();

        *out++ = '{';

        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            if (i != 0)
                std::format_to(out, ", ");

            out = std::format_to(out, "{}", vec[i]);
        }

        *out++ = '}';

        return out;
    }
};

template <typename ValueType>
bool operator==(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    if (lArg.size() != rArg.size())
        return false;

    for (std::size_t i = 0; i < lArg.size(); ++i)
        if (lArg[i] != rArg[i])
            return false;

    return true;
}

template <typename ValueType>
bool operator!=(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    return !(lArg == rArg);
}

template <typename ValueType>
auto operator*(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    if (lArg.size() != rArg.size())
        throw std::invalid_argument{"In order to do a dot product, the 2 vectors must be the same size"};

    ValueType output = 0;
    for (std::size_t i = 0; i < lArg.size(); ++i)
        output += lArg[i] * rArg[i];

    return output;
}

template <typename ValueType>
auto operator*(math_vector<ValueType> vec, ValueType scalar)
{
    return vec *= scalar;
}

template <typename ValueType>
auto operator*(ValueType scalar, math_vector<ValueType> vec)
{
    return vec *= scalar;
}

template <typename ValueType>
auto operator/(math_vector<ValueType> vec, ValueType scalar)
{
    return vec /= scalar;
}

template <typename ValueType>
auto operator+(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg += rArg;
}

template <typename ValueType>
auto operator-(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg -= rArg;
}

#endif //NNLEARNSQAGENT_MATH_VECTOR_H