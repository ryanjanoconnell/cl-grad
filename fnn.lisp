(in-package :cl-grad)

(defclass fnn-layer ()
  ((w :initarg :w
      :accessor w)
   (b :initarg :b
      :accessor b)
   (activation :initarg :activation
	       :accessor activation)))

(defun new-fnn-layer (input-units output-units activation)
  (make-instance
   'fnn-layer
   :w (random-tensor output-units input-units)
   :b (random-tensor output-units 1)
   :activation activation))

(defun feed-fnn-layer (x layer)
  (with-slots (w b activation) layer
    (funcall activation (add (mm w x) b))))

(defun new-fnn-model (activation &rest units)
  (let ((layers (make-array (1- (length units)))))
    (loop :for i :from 0 :to (- (length units) 2)
	  :do (let ((in (nth i units))
		    (out (nth (+ 1 i) units)))
		(setf (aref layers i) (new-fnn-layer in out activation))))
    layers))

(defun update-weight! (weight rate)
  (assert (grad weight))
  (sub! weight (scale! rate (grad weight))))

(defun update-fnn-layer-weights! (fnn-layer rate)
  (with-slots (w b) fnn-layer
    (update-weight! w rate)
    (update-weight! b rate)))

(defun feed (fnn-layers x)
  (reduce #'feed-fnn-layer fnn-layers :initial-value x))

(defun learn-noisy (fnn-layers xs ys &key (epoch 1) (rate 0.1))
  (print "STARTING TO LEARN")
  (dotimes (i epoch)
    (let ((count 0))
      (for-shuffle ((x xs) (y ys))
	(format t "Doing sample ~a~%" count)
	(incf count)
	(with-gradient-of (mse (feed fnn-layers x) y)
	  (for ((layer fnn-layers))
	    (update-fnn-layer-weights! layer rate)))))))

(defun learn (fnn-layers xs ys &key (epoch 1) (rate 0.1))
  (dotimes (i epoch)
    (for-shuffle ((x xs) (y ys))
      (with-gradient-of (mse (feed fnn-layers x) y)
	(for ((layer fnn-layers))
	  (update-fnn-layer-weights! layer rate))))))





