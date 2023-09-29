cl-grad is an implementation of reverse mode automatic differentiation for tensors. When tensor operations are performed, a computation graph is constucted implicitly. Then the function `backward` can be called on the terminal node in the computation graph to compute the gradient with respect to the terminal node of all the tensors in the graph. Here is an example with the operations used in feed forward neural networks. Suppose that `w` is an NxM matrix and `x`, `y`, `b` are Mx1 matrices.

```common-lisp
(let* ((affine     (add (mm w x) b))
       (activation (sigmoid affine))
       (cost       (mse activation y)))
  (backward cost)
  (print (grad w)))
```

The macro `defdiff` is used to define a new differentiable function. It must take in tensors and produce a new tensor without modifying its arguments for backprop to work. `defdiff` is a thin wrapper around `defun` that adds some metadata to the returned tensor (name of op, references to parents).

```common-lisp
(defdiff add (t1 t2)
  (entrywise #'+ t1 t2))
```
The macro `defgrad` is used to define the gradient functions of a differentiable function. It takes the numbe of arguments to the differentiable function plus an extra argument (the first in the parameter list) that corresponds to the upsteam node in the computation graph. The body of `defgrad` must be a function body that is used to compute the gradient of each of the inputs to the differentiable function. For example, to specify the gradient of the above function `add`:

```common-lisp
(defgrad add (t-out t1 t2)
  ;; gradient of t1
  (grad t-out)
  ;; gradient of t2
  (grad t-out))
```

Now `add` can be used to operate on tensors in a differentiable manner.

The composition of differentiable functions (which have a defined gradient) is also differentiable, without the need for specifying the gradient of the composition with a `defgrad`. For example, if `mul` is entrywise multiplication and has a corresponding `defgrad` then

```common-lisp
(defun new-fn (t1 t2)
  (mul (add t1 t2) t1))
```

will automatically be differentiable without the need to define the gradient with a `defgrad`.

The file `fnn.lisp` shows how cl-grad can be used to build a feed forward neural network in about 50 lines of code, and mnist.lisp applies a feed feed forward neural network to the MNIST dataset and achieves around 92% accuracy on the test set after a single epoch.