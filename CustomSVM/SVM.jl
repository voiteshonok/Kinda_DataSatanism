module svm

    using DataFrames
    using Random
    using LinearAlgebra
    using Zygote
    using Statistics
    using Flux
    using Flux: update! 
    import MLJ

    function linear_kernel(x_1, x_2)
        return x_1 * transpose(x_2)
    end

    Base.@kwdef mutable struct SVM
        lr::Float32 = 1e-3
        epochs::Int32=150
        lmbd::Float32 = 1e-4
        kernel_function = linear_kernel
        betas=nothing
        bias=nothing
        X=nothing
    end

    function predict(x, betas, bias)
        return linear_kernel(x, x) * transpose(betas) .+ bias 
    end

    function hinge_loss(scores, labels)
        return mean(relu(ones(size(scores)) .- dot(scores, labels))) # должны быть только не ноль но есть релу
    end

    function loss(x, y, betas, bias)
        preds = predict(x, betas, bias)
        return 0.01 * sum(betas * transpose(betas)) + hinge_loss(preds, y) # убрал K??
    end

    function fit(svm, X, Y)
        betas = rand(1, size(X, 1))
        bias = rand(1)

        opt = Flux.Adam(svm.lr)
        for epoch in 1:svm.epochs
            θ = Flux.params(betas, bias)
            gs = gradient(() -> loss(X, Y, betas, bias), θ)
            for p in (betas, bias)
                update!(opt, p, gs[p])
            end
        end
        svm.betas = betas
        svm.bias = bias
        svm.X = X
        return svm
    end

    function predict(svm, X)
        betas = svm.betas
        bias = svm.bias
        svm.kernel_function(svm.X, X) * transpose(betas) .+ bias
    end

    function get_accuracy(preds, label)
        answer = ones(size(preds)[1]) * (1)
        inds = findall(<(0), preds)
        answer[inds] .= 2
        mean(answer .== label)
    end

end