(defpackage :cl-grad
  (:use :cl))

(in-package :cl-grad)

(defclass tensor ()
  ((buffer :initarg :buffer
	   :accessor buffer)
   (dims :initarg :dims
	 :initform nil
	 :accessor dims)
   (op :initarg :op
       :initform nil
       :accessor op)
   (parents :initarg :parents
	    :initform nil
	    :accessor parents)
   (grad :initarg :grad
	 :initform nil
	 :accessor grad)))

(defvar *grad-fns* (make-hash-table))

(defun register-grad-fn (sym fn)
  (setf (gethash sym *grad-fns*) fn))

(defun get-grad-fn (sym)
  (let ((fn (gethash sym *grad-fns*)))
    (or fn (error "No registered grad-fn for ~a" sym))))

;; tensor utils
(defmethod print-object ((t1 tensor) s)
  (format s "<Tensor Dims: ~a Buffer: ~a>" (dims t1) (buffer t1)))

(defun ten (dims buffer)
  (assert (equal (reduce #'* dims) (length buffer)))
  (make-instance
   'tensor
   :dims (copy-seq dims)
   :buffer buffer))

(defun same-shape? (t1 t2)
  (equal (dims t1) (dims t2)))

(defun entry-count (t1)
  (reduce #'* (dims t1)))

(defun rank (t1)
  (length (dims t1)))

;;;;; Accessing and setting tensor values
(defun linearise-indices (ids dims)
  (assert (equal (length ids) (length dims)))
  (let ((stride 1)
	(idx 0))
    (loop :for i :from (1- (length ids)) :downto 0
	  :do (progn
		(incf idx (* (nth i ids) stride))
		(setf stride (* stride (nth i dims)))))
    idx))

(defun list< (l1 l2)
  (assert (equal (length l1) (length l2)))
  (or (null l1)
      (and (< (car l1) (car l2))
	   (list< (cdr l1) (cdr l2)))))

(defun can-tref? (t1 indices)
  (and (equal (rank t1) (length indices))
       (list< indices (dims t1))))

(defun tref (t1 &rest indices)
  (assert (can-tref? t1 indices))
  (aref (buffer t1)
	(linearise-indices indices (dims t1))))

(defun (setf tref) (val t1 &rest indices)
  (assert (can-tref? t1 indices))
  (setf (aref (buffer t1)
	      (linearise-indices indices (dims t1)))
	val))

;;;;; Handy tensors
(defun tensor-with-val (dims val)
  (make-instance
   'tensor
   :dims (copy-seq dims)
   :buffer
   (make-array (reduce #'* dims) :initial-element val)))

(defun ones (dims)
  (tensor-with-val dims 1.0))

(defun zeros (dims)
  (tensor-with-val dims 0.0))

(defun random-tensor (&rest dims)
  (let* ((n (reduce #'* dims))
	 (buffer (make-array n)))
    (dotimes (i n)
      (setf (aref buffer i) (- 0.5 (random 1.0))))
    (make-instance
     'tensor
     :dims (copy-seq dims)
     :buffer buffer)))

;; Macros to make defining differentiable functions and their
;; gradient functions more convenient
(defmacro defdiff (name params &body body)
  (let ((result-tensor (gensym "Tensor-")))
    `(defun ,name ,params
       (let ((,result-tensor (progn ,@body)))
	 (setf (op ,result-tensor) ',name)
	 (setf (parents ,result-tensor) (list ,@params))
	 ,result-tensor))))

(defmacro defgrad (name (t-out . t-ins) &body grad-bodies)
  `(register-grad-fn
    ',name
    (lambda (,t-out)
      (destructuring-bind ,t-ins (parents ,t-out)
	,@(mapcar (lambda (t-in grad-body)
		    `(add-setf (grad ,t-in) ,grad-body))
		  t-ins
		  grad-bodies)))))

;; Applys op entry wise to each argument
(defun entrywise (op t1 &rest tensors)
  (assert (every (lambda (ten) (same-shape? t1 ten)) tensors))
  (let ((buffer (make-array (entry-count t1))))
    (apply #'map-into
	   buffer
	   op
	   (buffer t1)
	   (mapcar #'buffer tensors))
    (make-instance
     'tensor
     :dims   (dims t1)
     :buffer  buffer)))

(defun entrywise! (op t1 &rest tensors)
  (assert (every (lambda (ten) (same-shape? t1 ten)) tensors))
  (apply #'map-into
	 (buffer t1)
	 op
	 (buffer t1)
	 (mapcar #'buffer tensors))
  t1)

;;;;; addition
(defdiff add (t1 t2)
  (entrywise #'+ t1 t2))

(defun addn (t1 t2)
  (cond
    ((and (null t1) (null t2))
     (error "addn needs a non nil arg"))
    ((null t1)  t2)
    ((null t2)  t1)
    ((and t1 t2) (add t1 t2))))

(defmacro add-setf (t1 t2)
  `(setf ,t1 (addn ,t1 ,t2)))

(defgrad add (t-out t1 t2)
  (grad t-out)
  (grad t-out))

(defun add! (t1 t2)
  (entrywise! #'+ t1 t2))


;;;;; matrix matrix multiplication
(defun matrix? (t1)
  (equal (rank t1) 2))

(defun can-mm? (t1 t2)
  (and (matrix? t1)
       (matrix? t2)
       (equal (second (dims t1)) (first (dims t2)))))

(defdiff mm (t1 t2)
  (assert (can-mm? t1 t2))
  (let* ((rows (first (dims t1)))
	 (cols (second (dims t2)))
	 (n (* rows cols))
	 (buffer (make-array n))
	 (result (make-instance
		  'tensor
		  :dims (list rows cols)
		  :buffer buffer))
	 (dot-len (second (dims t1))))
    (dotimes (i rows result)
      (dotimes (j cols)
	(dotimes (k dot-len)
	  (incf (tref result i j)
		(* (tref t1 i k) (tref t2 k j))))))))

(defgrad mm (t-out t1 t2)
  (mm (grad t-out) (transpose t2))
  (mm (transpose t1) (grad t-out)))

;;;;; transpose
(defdiff transpose (t1)
  (assert (matrix? t1))
  (let* ((buffer (make-array (entry-count t1)))
	 (rows (second (dims t1)))
	 (cols (first (dims t1)))
	 (dims (list rows cols))
	 (result (make-instance
		  'tensor :dims dims :buffer buffer)))
    (dotimes (i rows result)
      (dotimes (j cols)
	(setf (tref result i j)
	      (tref t1 j i))))))


;;;;; entrywise multiplication
(defdiff mul (t1 t2)
  (entrywise #'* t1 t2))

;;;;; entrywise subtraction
(defdiff sub (t1 t2)
  (entrywise #'- t1 t2))

(defun sub! (t1 t2)
  (entrywise! #'- t1 t2))

;;;;; Sigmoid
(defun scalar-sigmoid (x)
  (/ 1.0 (+ 1.0 (exp (- x)))))

(defdiff sigmoid (t1)
  (entrywise #'scalar-sigmoid t1))

(defgrad sigmoid (t-out t1)
  (let ((ones (ones (dims t-out))))
    (mul (grad t-out)
	 (mul t-out (sub ones t-out)))))

;;;;; Scaling
(defun singleton (val)
  (make-instance
   'tensor
   :buffer (make-array 1 :initial-element val)))

(defun singleton? (t1)
  (equal (entry-count t1) 1))

(defun singleton-val (t1)
  (assert (singleton? t1))
  (aref (buffer t1) 0))

(defun scale (alpha t1)
  (entrywise (lambda (x) (* alpha x)) t1))

(defdiff scale% (t1 t2)
  (assert (singleton? t1))
  (entrywise (lambda (entry) (* (singleton-val t1) entry))
	     t2))

(defun scale! (alpha t1)
  (entrywise! (lambda (x) (* alpha x)) t1))

;;;;; Mean squared error
(defdiff mse (t1 t2)
  (assert (same-shape? t1 t2))
  (let* ((n (entry-count t1))
	 (it (make-array n)))
    (map-into it
	      (lambda (x y)
		(expt (- x y) 2))
	      (buffer t1)
	      (buffer t2))
    (setf it (reduce #'+ it))
    (setf it (/ it n))
    (singleton it)))

(defgrad mse (t-out t1 t2)
  (scale% t-out (sub t1 t2))
  (scale% t-out (sub t2 t1)))

;; Backprop
(defun detach-all-grads (t1)
  (setf (grad t1) nil)
  (when (parents t1)
    (mapc #'detach-all-grads (parents t1))))

(defun compute-parent-grads (t1)
  (funcall (get-grad-fn (op t1)) t1))

(defun backward (t1)
  ;; Only compute gradients wrt a scalar
  (assert (singleton? t1))
  ;; ensure no lingering gradients in the graph
  (detach-all-grads t1)
  ;; backpropagation
  (setf (grad t1) (singleton 1.0))
  (labels ((backprop (tn)
	     (when (parents tn)
	       (compute-parent-grads tn)
	       (mapc #'backprop (parents tn)))))
    (backprop t1)
    nil))

(defmacro with-gradient-of (cost &body body)
  `(progn
     (backward ,cost)
     ,@body
     (detach-all-grads ,cost)))

