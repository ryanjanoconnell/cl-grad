(in-package :cl-user)

(defpackage :cl-grad
  (:use :cl))

(in-package :cl-grad)

(defclass tensor ()
  ((buffer :initarg :buffer
	   :accessor buffer)
   (dims :initarg :dims
	 :initform nil
	 :accessor dims)))

(defclass var ()
  ((tensor :initarg :tensor
	   :accessor tensor)
   (op :initarg :op
       :initform nil
       :accessor op)
   (parents :initarg :parents
	    :initform nil
	    :accessor parents)
   (grad :initarg :grad
	 :initform nil
	 :accessor grad)))

;; Table of gradient functions
(defvar *grad-fns* (make-hash-table))

(defun register-grad-fn (sym fn)
  (setf (gethash sym *grad-fns*) fn))

(defun get-grad-fn (sym)
  (let ((fn (gethash sym *grad-fns*)))
    (or fn (error "No registered grad-fn for ~a" sym))))

;; utils
(defun copy-tensor (t1)
  (make-instance
   'tensor
   :dims (copy-list (dims t1))
   :buffer (copy-seq (buffer t1))))

(defun copy-var (v)
  (make-instance
   'var
   :parents (copy-list (parents v))
   :op (op v)
   :tensor (copy-tensor (tensor v))
   :grad (grad v)))

(defun var (dims data)
  (assert (equal (reduce #'* dims) (length data)))
  (make-instance
   'var
   :tensor (make-instance
	    'tensor
	    :dims (copy-list dims)
	    :buffer (copy-seq data))))

(defmethod print-object ((t1 tensor) s)
  (format s "<Tensor Dims: ~a Buffer: ~a>" (dims t1) (buffer t1)))

(defmethod print-object ((v var) s)
  (format s "<Var Dims: ~a Data: ~a>" (dims (tensor v)) (buffer (tensor v))))

(defun same-shape? (t1 t2)
  (equal (dims t1) (dims t2)))

(defun matrix? (t1)
  (equal (length (dims t1)) 2))

(defun scalar? (t1)
  (equal (entry-count t1) 1))

(defun rows (t1)
  (first (dims t1)))

(defun cols (t1)
  (second (dims t1)))

(defun entry-count (t1)
  (reduce #'* (dims t1)))

(defun can-matmul? (t1 t2)
  (and (matrix? t1)
       (matrix? t2)
       (equal (cols t1) (rows t2))))

(defun linearise-indices (ids dims)
  (assert (equal (length ids) (length dims)))
  (let ((stride 1)
	(idx 0))
    (loop :for i :from (1- (length ids)) :downto 0
	  :do (progn
		(incf idx (* (nth i ids) stride))
		(setf stride (* stride (nth i dims)))))
    idx))

(defmacro tref (t1 &rest indices)
  `(aref (buffer ,t1)
	 (linearise-indices (list ,@indices) (dims ,t1))))

(defun tensor->var (t1)
  (make-instance 'var :tensor t1))

(defun scalar-val (t1)
  (assert (scalar? t1))
  (aref (buffer t1) 0))

(defun scalar (val)
  (tensor->var (make-instance
		'tensor
		:buffer
		(make-array 1 :initial-element val))))


;; Generators
(defun t-ones (dims)
  (make-instance
   'tensor
   :dims (copy-list dims)
   :buffer
   (make-array (reduce #'* dims) :initial-element 1.0)))

(defun ones (dims)
  (tensor->var (t-ones dims)))

;; tensor entrywise function maker
(defun entrywise-tensor-fn (op t1 &rest ts)
  (assert (every (lambda (tn) (same-shape? t1 tn)) ts))
  (let ((result	(make-instance
		 'tensor
		 :dims (copy-list (dims t1))
		 :buffer (make-array (reduce #'* (dims t1))))))
    (dotimes (i (length (buffer result)) result)
      (setf (aref (buffer result) i)
	    (apply op
		   (aref (buffer t1) i)
		   (mapcar (lambda (tn) (aref (buffer tn) i))
			   ts))))))

;; Lift a tensor-fn to a var-fn
(defmacro define-var-fn (name t-fn)
  `(defun ,name (&rest vars)
     (let ((ts (mapcar #'tensor vars)))
       (make-instance
	'var
	:parents vars
	:op ',name
	:tensor (apply ,t-fn ts)))))

;; Define a gradient function for a var function
(defmacro define-grad-fn (var-fn-name params &body grad-bodies)
  (assert (equal (1- (length  params)) (length grad-bodies)))
  (let ((v-out (car params))
	(v-ins (cdr params)))
    `(register-grad-fn
      ',var-fn-name
      (lambda (,v-out)
	(destructuring-bind ,v-ins (parents ,v-out)
	  ,@(mapcar (lambda (v-in grad-body)
		      `(add-setf (grad ,v-in) ,grad-body))
		    v-ins
		    grad-bodies))))))

;; Scale
(defun t-scale (t1 t2)
  (assert (scalar? t1))
  (let ((result (copy-tensor t2)))
    (dotimes (i (length (buffer result)) result)
      (setf (aref (buffer result) i)
	    (* (scalar-val t1)
	       (aref (buffer result) i))))))

(define-var-fn scale #'t-scale)

;; Add
(defun t-add (t1 t2)
  (entrywise-tensor-fn #'+ t1 t2))

(define-var-fn add #'t-add)

(defun addn (v1 v2)
  (cond
    ((and (null v1) (null v2))
     (error "addn needs a non nil arg"))
    ((null v1) (copy-var v2))
    ((null v2) (copy-var v1))
    ((and v1 v2) (add v1 v2))))

(defmacro add-setf (t1 t2)
  `(setf ,t1 (addn ,t1 ,t2)))

(define-grad-fn add (v-out v1 v2)
  (grad v-out)
  (grad v-out))

;; Subtract
(defun t-sub (t1 t2)
  (entrywise-tensor-fn #'- t1 t2))

(define-var-fn sub #'t-sub)

;; Multiplication
(defun t-mul (t1 t2)
  (entrywise-tensor-fn #'* t1 t2))

(define-var-fn mul #'t-mul)

;; Division
(defun t-div (t1 t2)
  (entrywise-tensor-fn #'/ t1 t2))

(define-var-fn div #'t-div)

;; Transpose
(defun t-transpose (t1)
  (assert (matrix? t1))
  (let ((result (make-instance
		 'tensor
		 :dims (list (cols t1) (rows t1))
		 :buffer (make-array (* (cols t1) (rows t1))))))
    (dotimes (i (rows result) result)
      (dotimes (j (cols result))
	(setf (tref result i j) (tref t1 j i))))))

(define-var-fn transpose #'t-transpose)

;; Matmul
(defun t-matmul (t1 t2)
  (assert (can-matmul? t1 t2))
  (let ((result (make-instance
		 'tensor
		 :dims (list (rows t1) (cols t2))
		 :buffer (make-array (* (rows t1) (cols t2))
				     :initial-element 0.0))))
    (dotimes (i (rows result) result)
      (dotimes (j (cols result))
	(dotimes (k (cols t1))
	  (incf (tref result i j)
		(* (tref t1 i k) (tref t2 k j))))))))

(define-var-fn matmul #'t-matmul)

(define-grad-fn matmul (v-out v1 v2)
  (matmul (grad v-out) (transpose v2))
  (matmul (transpose v1) (grad v-out)))

;; Sigmoid
(defun sig (x)
  (/ 1 (+ 1 (exp (- x)))))

(defun t-sigmoid (t1)
  (entrywise-tensor-fn #'sig t1))

(define-var-fn sigmoid #'t-sigmoid)

(defun dsig (v)
  (let ((ones (ones (dims (tensor v)))))
    (mul v (sub ones v))))

(define-grad-fn sigmoid (v-out v)
  (mul (grad v-out) (dsig v-out)))

;; MSE loss
(defun t-mse (t1 t2)
  (assert (same-shape? t1 t2))
  (let ((result (make-instance
		 'tensor
		 :buffer
		 (make-array 1)))
	(n (entry-count t1))
	(acc 0.0))
    (dotimes (i n)
      (incf acc
	    (expt (- (aref (buffer t1) i)
		     (aref (buffer t2) i))
		  2)))
    (setf (aref (buffer result) 0)
	  (/ acc n))
    result))

(define-var-fn mse #'t-mse)

(define-grad-fn mse (v-out v1 v2)
  (scale (mul (grad v-out)
	      (scalar (/ 1 (entry-count (tensor v-out)))))
	 (sub v1 v2))
  (scale (mul (grad v-out)
	      (scalar (/ 1 (entry-count (tensor v-out)))))
	 (sub v2 v1)))

;; Backprop
(defun compute-parent-grads (v)
  (funcall (get-grad-fn (op v)) v))

(defun backward (v)
  (assert (scalar? (tensor v)))
  (setf (grad v) (scalar 1.0))
  (labels ((recur (var)
	     (when (parents var)
	       (compute-parent-grads var)
	       (mapc #'recur (parents var)))))
    (recur v)
    nil))


;;;;; This outputs the same as pytorch ;;;;;;
;; (let* ((w (var '(2 2) #(0.1 0.2 0.3 0.4)))
;;        (b (var '(2 1) #(0.2 0.3)))
;;        (x (var '(2 1) #(0.5 0.1)))
;;        (y (var '(2 1) #(0.1 0.2)))
;;        (affine (add (matmul w x) b))
;;        (sig (sigmoid affine))
;;        (cost (mse sig y)))
;;   (backward cost)
;;   (format t "Gradients: ~%")
;;   (format t "Dw = ~a~%" (grad w))
;;   (format t "Db = ~a~%" (grad b))
;;   (format t "Dx = ~a~%" (grad x))
;;   (format t "Dy = ~a~%" (grad y)))


