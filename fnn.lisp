(eval-when (:compile-toplevel :load-toplevel :execute)
  (load "./cl-grad.lisp"))

(in-package :cl-grad)

(defclass fnn ()
  ((weights :initarg :weights
	    :accessor weights)
   (biases :initarg :biases
	   :accessor biases)))

(defmethod print-object ((fnn fnn) s)
  (format s "~%FNN:~%")
  (dotimes (i (length (weights fnn)))
    (format s "Layer ~a:~%" (1+ i))
    (format s "~a~%" (aref (weights fnn) i))
    (format s "~a~%" (aref (biases fnn) i))
    (format s "~%")))

(defun new-fnn (&rest widths)
  (let* ((n (1- (length widths)))
	 (ws (make-array n))
	 (bs (make-array n))
	 (result (make-instance 'fnn :biases bs :weights ws)))
    (dotimes (i n result)
      (let ((in-width (nth i widths))
	    (out-width (nth (1+ i) widths)))
	(setf (aref ws i) (rand-matrix out-width in-width))
	(setf (aref bs i) (rand-matrix out-width 1))))))

(defun feed (fnn input)
  (let ((x input))
    (loop :for w :across (weights fnn)
	  :for b :across (biases fnn)
	  :do (progn
		(setf x (t-sigmoid (t-add (t-matmul w x) b)))))
    x))

(defun shuffle (vec)
  (let ((temp)
	(j))
    (loop :for i :from (1- (length vec)) :downto 1
	  :do (progn
		(setf j (random (+ i 1)))
		(setf temp (aref vec i))
		(setf (aref vec i) (aref vec j))
		(setf (aref vec j) temp)))
    vec))

(defun range (n)
  (let ((vec (make-array n)))
    (dotimes (i n vec)
      (setf (aref vec i) i))))

(defun update-param! (t1 grad-t1 rate)
  (t-add! t1 (t-scale! (* -1.0 rate) grad-t1)))

(defparameter *noisy* t)
(defparameter *rate* 0.1)
(defparameter *epoch* 100)

;; x, y are tensors
(defun learn-1 (fnn x y)
  ;; Wrap everything in a Var
  (let* ((vx (tensor->var x))
	 (vy (tensor->var y))
	 (vws (map 'vector #'tensor->var (weights fnn)))
	 (vbs (map 'vector #'tensor->var (biases fnn)))
	 (cost))
    ;; Compute Cost
    (loop :for vw :across vws
	  :for vb :across vbs
	  :do (setf vx (sigmoid (add (matmul vw vx) vb))))
    (setf cost (mse vx vy))
    ;; Compute Gradient of Cost
    (backward cost)
    ;; Update Weights and Biases
    (loop :for vw :across vws
	  :for vb :across vbs
	  :do (progn
		(update-param! (tensor vw) (tensor (grad vw)) *rate*)
		(update-param! (tensor vb) (tensor (grad vb)) *rate*)))))

;; x-train, y-train are vectors of tensors
(defun learn (fnn x-train y-train)
  (assert (equal (length x-train) (length y-train)))
  (let* ((n (length x-train))
	 (ids (range n)))
    (dotimes (epoch *epoch* fnn)
      (shuffle ids)
      (loop :for i :across ids
	    :for count :from 0 :to (1- n)
	    :do (let* ((x (aref x-train i))
		       (y (aref y-train i)))
		  (when (and *noisy* (equal (mod count 1000) 0)) 
		    (format t "Doing sample ~a~%" count))
		  (learn-1 fnn x y))))))





