(in-package :cl-user)

(defpackage :cl-grad
  (:use :cl))

(in-package :cl-grad)

(defclass tensor ()
  (;; vector of floats
   (data :initarg :data
	 :accessor data)
   ;; list of dims
   (dims :initarg :dims
	 :accessor dims)
   ;; keyword indicating op that produced tensor
   (op :initarg :op
       :initform nil
       :accessor op)
   ;; list of arguments to the op
   (parents :initarg :parents
	    :initform nil
	    :accessor parents)
   ;; pointer to gradient tensor
   (grad :initarg :grad
	 :initform nil
	 :accessor grad)))

;; utils
(defun same-shape? (t1 t2)
  (equal (dims t1) (dims t2)))

(defun entry-count (t1)
  (reduce #'* (dims t1)))

(defun matrix? (t1)
  (equal (length (dims t1)) 2))

(defun rows (t1)
  (first (dims t1)))

(defun cols (t1)
  (second (dims t1)))

(defun scalar? (t1)
  (null (dims t1)))

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
  `(aref (data ,t1)
	 (linearise-indices (list ,@indices) (dims ,t1))))

(defun copy-tensor (t1)
  (make-instance 'tensor
		 :dims (copy-list (dims t1))
		 :parents (copy-list (parents t1))
		 :op (op t1)
		 :grad (grad t1)
		 :data (copy-seq (data t1))))

(defun ones (dims)
  (make-instance 'tensor
		 :dims (copy-list dims)
		 :data
		 (make-array (reduce #'* dims) :initial-element 1.0)))


(defun ten (dims &rest entries)
  (let ((n (reduce #'* dims)))
    (assert (equal n (length entries)))
    (let ((res (make-instance 'tensor
			      :dims dims
			      :data (make-array n))))
      (dotimes (i n res)
	(setf (aref (data res) i)
	      (nth i entries))))))

(defmethod print-object ((ten tensor) stream)
  (format stream
	  "<~aD-Tensor dims: ~a data: ~a>" (length (dims ten)) (dims ten) (data ten)))

;; entry-wise addition
(defun add-prepare-child (t1 t2)
  (assert (same-shape? t1 t2))
  (make-instance 'tensor
		 :parents (list t1 t2)
		 :op :add
		 :dims (copy-list (dims t1))
		 :data (make-array (reduce #'* (dims t1)))))

(defun add-compute (t1)
  (let ((p1 (first (parents t1)))
	(p2 (second (parents t1))))
    (dotimes (i (length (data t1)))
      (setf (aref (data t1) i)
	    (+ (aref (data p1) i)
	       (aref (data p2) i)))))
  t1)

(defun add (t1 t2)
  (if t1
      (let ((result (add-prepare-child t1 t2)))
	(add-compute result))
      (copy-tensor t2)))

;; entry-wise multiplication
(defun mul-prepare-child (t1 t2)
  (assert (same-shape? t1 t2))
  (make-instance 'tensor
		 :parents (list t1 t2)
		 :op :mul
		 :dims (copy-list (dims t1))
		 :data (make-array (reduce #'* (dims t1)))))

(defun mul-compute (t1)
  (destructuring-bind (p1 p2) (parents t1)
    (dotimes (i (length (data t1)))
      (setf (aref (data t1) i)
	    (* (aref (data p1) i) (aref (data p2) i))))
    t1))

(defun mul (t1 t2)
  (let ((result (mul-prepare-child t1 t2)))
    (mul-compute result)))

;; entry-wise subtract
;; entry-wise multiplication
(defun sub-prepare-child (t1 t2)
  (assert (same-shape? t1 t2))
  (make-instance 'tensor
		 :parents (list t1 t2)
		 :op :sub
		 :dims (copy-list (dims t1))
		 :data (make-array (reduce #'* (dims t1)))))

(defun sub-compute (t1)
  (destructuring-bind (p1 p2) (parents t1)
    (dotimes (i (length (data t1)))
      (setf (aref (data t1) i)
	    (- (aref (data p1) i) (aref (data p2) i))))
    t1))

(defun sub (t1 t2)
  (let ((result (sub-prepare-child t1 t2)))
    (sub-compute result)))

;; Matrix multiplication
(defun matmul-prepare-child (t1 t2)
  (assert (can-matmul? t1 t2))
  (make-instance 'tensor
		 :parents (list t1 t2)
		 :op :matmul
		 :dims (list (rows t1) (cols t2))
		 :data (make-array (* (rows t1) (cols t2)))))

(defun matmul-compute (t1)
  (let ((p1 (first (parents t1)))
	(p2 (second (parents t1))))
    (fill (data t1) 0)
    (dotimes (i (rows t1))
      (dotimes (j (cols t1))
	(dotimes (k (cols p1))
	  (incf (tref t1 i j)
		(* (tref p1 i k) (tref p2 k j))))))
    t1))

(defun matmul (t1 t2)
  (let ((result (matmul-prepare-child t1 t2)))
    (matmul-compute result)))

;; scale

(defun scalar (val)
  (make-instance 'tensor
		 :dims nil
		 :data (make-array 1 :initial-element val)))

(defun scale (t1 t2)
  (assert (scalar? t1))
  (let ((result (copy-tensor t2)))
    (dotimes (i (length (data result)) result)
      (setf (aref (data result) i)
	    (* (aref (data result) i)
	       (aref (data t1) 0))))))

(defun div (t1 t2)
  (assert (scalar? t1))
  (let ((result (copy-tensor t2)))
    (dotimes (i (length (data result)) result)
      (setf (aref (data result) i)
	    (/ (aref (data result) i)
	       (aref (data t1) 0))))))

(defun translate (t1 t2)
  (assert (scalar? t1))
  (let ((result (copy-tensor t2)))
    (dotimes (i (length (data result)) result)
      (setf (aref (data result) i)
	    (+ (aref (data result) i)
	       (aref (data t1) 0))))))

;; Sigmoid
(defun sigmoid (x)
  (/ 1 (+ 1 (exp (- x)))))

(defun sig-prepare-child (t1)
  (make-instance 'tensor
		 :parents (list t1)
		 :op :sig
		 :dims (copy-list (dims t1))
		 :data (make-array (reduce #'* (dims t1)))))

(defun sig-compute (t1)
  (let ((p1 (first (parents t1))))
    (dotimes (i (length (data t1)))
      (setf (aref (data t1) i)
	    (sigmoid (aref (data p1) i))))
    t1))

(defun sig (t1)
  (let ((result (sig-prepare-child t1)))
    (sig-compute result)))

(defun dsig (t1)
  (mul t1 (sub (ones (dims t1)) t1)))

;; Mean Squared Error
(defun mse-prepare-child (t1 t2)
  (assert (same-shape? t1 t2))
  (make-instance 'tensor
		 :parents (list t1 t2)
		 :op :mse
		 :dims nil
		 :data (make-array 1)))

(defun mse-compute (t1)
  (let* ((p1 (first (parents t1)))
	 (p2 (second (parents t1)))
	 (n (length (data p1)))
	 (acc 0))
    (dotimes (i n)
      (incf acc (expt (- (aref (data p1) i) (aref (data p2) i)) 2)))
    (setf (aref (data t1) 0) (/ acc n))
    t1))

(defun mse (t1 t2)
  (let ((result (mse-prepare-child t1 t2)))
    (mse-compute result)))

;; Matrix transpose
(defun transpose-prepare-child (t1)
  (assert (matrix? t1))
  (make-instance 'tensor
		 :parents (list t1)
		 :op :transpose
		 :dims (list (cols t1) (rows t1))
		 :data (make-array (* (cols t1) (rows t1)))))

(defun transpose-compute (t1)
  (let ((p1 (first (parents t1))))
    (dotimes (i (rows p1))
      (dotimes (j (cols p1))
	(setf (tref t1 j i) (tref p1 i j))))
    t1))

(defun transpose (t1)
  (let ((result (transpose-prepare-child t1)))
    (transpose-compute result)))

;; Backprop
(defun compute-parent-grads (t1)
  (destructuring-bind (p1 &optional p2) (parents t1)
    (ecase (op t1)
      (:add
       (setf (grad p1)
	     (add (grad p1) (grad t1)))
       (setf (grad p2)
	     (add (grad p2) (grad t1))))
      
      (:matmul
       (setf (grad p1)
	     (add (grad p1) (matmul (grad t1) (transpose p2))))
       (setf (grad p2)
	     (add (grad p2) (matmul (transpose p1) (grad t1)))))
      
      (:sig
       (setf (grad p1)
	     (add (grad p1) (mul (grad t1) (dsig t1)))))
      
      (:mse
       (setf (grad p1)
	     (add (grad p1)
		  (scale (mul (grad t1) (scalar (/ 2 (entry-count p1))))
			 (sub p1 p2))))
       (setf (grad p2)
	     (add (grad p2)
		  (scale (mul (grad t1) (scalar (/ 2 (entry-count p2))))
			 (sub p2 p1))))))))

(defun backprop (t1)
  (assert (scalar? t1))
  (setf (grad t1) (scalar 1.0))
  (labels ((recur (tn)
	     (when (parents tn)
	       (compute-parent-grads tn)
	       (mapc #'recur (parents tn)))))
    (recur t1)))


;; sanity tests, comparing outputs to pytorch
;; it works for this example

;; (let*
;;     ((w (ten '(2 2)
;; 	     0.1 0.2
;; 	     0.3 0.4))
       
;;      (b (ten '(2 1)
;; 	     0.2
;; 	     0.3))
       
;;      (x (ten '(2 1)
;; 	     0.5
;; 	     0.1))
       
;;      (y (ten '(2 1)
;; 	     0.1
;; 	     0.2))
     
;;      (affine (add (matmul w x) b))
;;      (sig (sig affine))
;;      (c (mse sig y)))
;;   (backprop c)
;;   (format t "~%Gadients:~%~%")
;;   (format t "dc/dc = ~a~%~%" (grad c))
;;   (format t "dc/dy = ~a~%~%" (grad y))
;;   (format t "dc/dw = ~a~%~%" (grad w))
;;   (format t "dc/db = ~a~%~%" (grad b))
;;   (format t "dc/dx = ~a~%~%" (grad x)))

