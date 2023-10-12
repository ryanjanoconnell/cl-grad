cl-grad is an implementation of reverse mode automatic differentiation for tensors. When tensor operations are performed, a computation graph is constucted. Then the function `backward` can be called to computate the of the terminal node with respect to the inputs. Here is an example with the operations used in feed forward neural networks. Suppose that `w` is an NxM matrix and `x`, `y`, `b` are Mx1 matrices.

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
The macro `defgrad` is used to define the gradient functions of a differentiable function. It takes the same number of arguments as the differentiable function plus an extra argument (the first in the parameter list) that corresponds to the upsteam node in the computation graph. The body of `defgrad` must be a set of function bodies that are used to compute the gradient for each of the inputs to the differentiable function. For example, to specify the gradient of the above function `add`:

```common-lisp
(defgrad add (t-out t1 t2)
  ;; gradient of t1
  (grad t-out)
  ;; gradient of t2
  (grad t-out))
```

Now `add` can be used to operate on tensors in a differentiable manner.

The file `fnn.lisp` shows how cl-grad can be used to build a feed forward neural network in about 50 lines of code, and mnist.lisp applies a feed feed forward neural network to the MNIST dataset and achieves around 92% accuracy on the test set after a single epoch.