(load "./cl-grad.lisp")

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

(defun rand-matrix (rows cols)
  (let ((result-tensor (make-instance
			'tensor
			:dims (list rows cols)
			:buffer (make-array (* rows cols)))))
    (dotimes (i (length (buffer result-tensor)))
      (setf (aref (buffer result-tensor) i) (random 0.1)))
    (tensor->var result-tensor)))

(defun new-fnn (&rest dims)
  (let* ((n (1- (length dims)))
	 (ws (make-array n))
	 (bs (make-array n))
	 (result (make-instance 'fnn :biases bs :weights ws)))
    (dotimes (i n result)
      (let ((in-dim (nth i dims))
	    (out-dim (nth (1+ i) dims)))
	(setf (aref ws i) (rand-matrix out-dim in-dim))
	(setf (aref bs i) (rand-matrix out-dim 1))))))

(defun feed (fnn input)
  (let ((x input))
    (loop :for w :across (weights fnn)
	  :for b :across (biases fnn)
	  :do (setf x (sigmoid (add (matmul w x) b))))
    x))

(defun add! (v1 v2)
  "Adds the adds the buffer of v2 into the buffer of v1,
   bypassing the computational graph logic"
  (let ((t1 (tensor v1))
	(t2 (tensor v2)))
    (dotimes (i (length (buffer t1)) t1)
      (incf (aref (buffer t1) i)
	    (aref (buffer t2) i)))))

(defun update-weights! (fnn rate)
  (loop :for w :across (weights fnn)
	:for b :across (biases fnn)
	:do (progn
	      (add! w (scale (scalar (- rate)) (grad w)))
	      (add! b (scale (scalar (- rate)) (grad b))))))

(defun cost (fnn x y)
  (mse (feed fnn x) y))

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

(defun item (v)
  (assert (scalar? (tensor v)))
  (aref (buffer (tensor v)) 0))

(defparameter *noisy* t)
(defparameter *rate* 0.5)
(defparameter *epoch* 100)

;; x-train, y-train are vectors of vars
(defun learn (fnn x-train y-train)
  (assert (equal (length x-train) (length y-train)))
  (let* ((n (length x-train))
	 (ids (range n)))
    (dotimes (epoch *epoch* fnn)
      (shuffle ids)
      (loop :for i :across ids
	    :do (let* ((x (aref x-train i))
		       (y (aref y-train i))
		       (c (cost fnn x y)))
		  (when *noisy* (format t "C = ~a~%" (item c)))
		  (backward c)
		  (update-weights! fnn *rate*))))))


;; XOR example seems to work occassionally.
;; Alot of float overflows due to sigmoid though
;;
;;
;; (let ((x-train) (y-train) (fnn))
;;  
;;   (setf fnn (new-fnn 2 2 1))
;;  
;;   (setf x-train
;; 	(vector (var '(2 1) #(0 0))
;; 		(var '(2 1) #(1 0))
;; 		(var '(2 1) #(0 1))
;; 		(var '(2 1) #(1 1))))
;;  
;;   (setf y-train
;; 	(vector (var '(1 1) #(0))
;; 		(var '(1 1) #(1))
;; 		(var '(1 1) #(1))
;; 		(var '(1 1) #(0))))
;;
;;   (setf *noisy* t)
;;  
;;   (learn fnn x-train y-train))





