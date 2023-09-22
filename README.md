This is a tensor automatic differentiation library. It is an implementation of reverse mode automatic differentiation that I indend to use to for a deep learning library at some stage.

The tensor class represents a view into an array. The var class represents a node in a tensor computation graph. This library keeps separate the concept of a node in the computation graph and the underlying tensor. A tensor op can be defined as a function that takes in tensors and returns a new tensor. Tensor ops can be made differentiable using the macro `define-var-op` to lift the tensor op to a var op (a function which takes in vars and returns a var) and then `define-grad-fn` to specify how the gradient of a var op should be computed.

`define-var-op` takes in the name of the var op you wish to define and the name of the underlying tensor op. `define-grad-fn` takes in the name of the var op you are defining gradients for, a parameter list and the recipes for computing the gradients of the arguments to the var op.

After a computation graph has been built by applying var ops to vars, the gradients can be computed using `(backward terminal-var)` where `terminal-var` is the terminal nod ein the computation graph.