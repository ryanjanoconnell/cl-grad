cl-grad is an automatic differentiation library for tensors. It provides facilities to make tensor functions differentiable via reverse mode automatic differentiation. Tensors are wrapped in a Var class which helps to build a computation graph as tensors are operated on. To compute gradients with respect to a scalar output,  the computation graph is traversed in reverse and Jacobian products are computed along the way.

Given a tensor function (a function which takes in tensors and outputs a new tensor), the `define-var-fn` macro can be used to construct a new function which uses the given tensor function under the hood, but adds a node to the computation graph whenever it is called. For example, if `(tensor-add t1 t2)` takes in two tensors `t1`, `t2` and outputs a new tensor which is the entrywise sum of `t1`, `t2` then

```common-lisp
(define-var-fn var-add #'tensor-add)
```

will define a function called `var-add` which acts the same as `tensor-add` but adds a node to the computation graph each time it is called.

The `define-grad-fn` macro specifies how the gradient for a var function should be computed. For example,

```common-lisp
(define-grad-fn var-add (v-out v1 v2)
  (grad v-out)
  (grad v-out))
```

defines the recipe for computing the gradient of a `var-add` node with respect to some upstream scalar output.

The file fnn.lisp shows how cl-grad can be used to define feed forward neural networks. The file mnist.lisp uses the defined feed forward neural network to train an image classifier that achieves 92% accuracy on the test data.

