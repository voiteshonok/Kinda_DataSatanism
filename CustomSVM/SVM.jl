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
        lmbd::Float32 = 1e-2
        kernel_function = linear_kernel
        betas=nothing
        bias=nothing
        X=nothing
    end

    function predict(x, svm)
        return linear_kernel(x, x) * transpose(svm.betas) .+ svm.bias 
    end

    function hinge_loss(scores, labels)
        return mean(relu(ones(size(scores)) .- dot(scores, labels)))
    end

    function loss(x, y, svm)
        preds = predict(x, svm)
        return svm.lmbd * sum(svm.betas * transpose(svm.betas)) + hinge_loss(preds, y)
    end

    function fit(svm, X, Y)
        svm.X = X
        svm.betas = rand(1, size(X, 1))
        svm.bias = rand(1)

        opt = Flux.Adam(svm.lr)
        for epoch in 1:svm.epochs
            θ = Flux.params(svm.betas, svm.bias)
            gs = gradient(() -> loss(X, Y, svm), θ)
            for p in (svm.betas, svm.bias)
                update!(opt, p, gs[p])
            end
        end
    
        return svm
    end

    function predict_score(svm, X)
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