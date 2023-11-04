using Parameters
using LinearAlgebra
using JSON
using Serialization

import Parameters.with_kw
include("modify.jl")

@with_kw mutable struct Weights
    w1::Matrix
    b1::Vector

    w2::Vector
    b2::Number
end

nodes_in_hidden_layer = 2
inputs = 2

function fill_weights(f)::Weights
    return Weights(
        w1 = (nodes_in_hidden_layer, inputs) |> f,
        b1 = nodes_in_hidden_layer |> f,

        w2 = nodes_in_hidden_layer |> f,
        b2 = f([1])[1]
    )
end

@with_kw struct Output
    z1::Vector
    a1::Vector

    z2::Number
    a2::Number
end

function learn!(weights::Weights, dataset, iterations::Integer, rate::Number)
    for _ in 1:iterations
        gradient = compute_cost_gradient(weights, dataset)
        update_weights!(weights, gradient, rate)
    end
end

function update_weights!(weights::Weights, gradient::Weights, rate::Number)
    modify!((w, g) -> w - rate * g, weights, gradient)
end

function compute_cost_gradient(weights::Weights, dataset)::Weights
    gradient = fill_weights(dims -> fill(Float32(0), dims...))

    for (x, y) in dataset
        output = compute_output(weights, x)
        derivatives = compute_derivatives(weights, output, x, y)

        modify!(+, gradient, derivatives)
    end

    modify!(/, gradient, length(dataset))

    return gradient
end

function compute_output(weights::Weights, x::Vector)::Output
    z1 = weights.w1 * x + weights.b1
    a1 = sigmoid.(z1)

    z2 = dot(weights.w2, a1) + weights.b2
    a2 = sigmoid(z2)

    return Output(z1, a1, z2, a2)
end

function compute_derivatives(weights::Weights, output::Output, x::Vector, y::Number)::Weights
    common = -2 * (y - output.a2) * sigmoid_prime_from_sigmoid(output.a2)

    w2 = common * output.a1

    common1 = @. common * weights.w2 * sigmoid_prime_from_sigmoid(output.a1)
    w1 = common1 .* transpose(x)

    b2 = common
    b1 = common1

    return Weights(w1, b1, w2, b2)
end

function sigmoid(x)
    1 / (1 + exp(-x))
end

function sigmoid_prime_from_sigmoid(x)
    x * (1 - x)
end

function test(weights::Weights, dataset)
    cost = 0

    for (x, y) in dataset
        output = compute_output(weights, x)
        cost += (y - output.a2)^2

        println("Expected: $y. Got $(output.a2)")
    end

    cost /= length(dataset)
    
    println("Cost: $cost")
end

function train_and_test_and_save(dataset)
    weights = fill_weights(dims -> randn(Float32, dims...))

    iterations = 100
    rate = 10
    
    learn!(weights, dataset, iterations, rate)

    println("Rate: $rate. Iterations: $iterations")
    test(weights, dataset)

    open("weights.bin", "w") do f
        serialize(f, weights)
    end

    open("readable-weights.json", "w") do f
        JSON.print(f, weights)
    end
end

function load_and_test(dataset)
    weights::Weights = deserialize("weights.bin")

    print(compute_output(weights, [1, 0]))
    print(compute_output(weights, [0, 1]))

    # test(weights, dataset)
end

dataset = [
    (x=[0, 0], y = 0),
    (x=[0, 1], y = 1),
    (x=[1, 0], y = 1),
    (x=[1, 1], y = 0)
]

#train_and_test_and_save(dataset)
load_and_test(dataset)
