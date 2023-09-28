(in-package :cl-grad)

(defun bytes->int (byte-vec)
  "Converts an array of bytes in big-endian to an integer"
  (let ((n (1- (length byte-vec)))
	(res 0))
    (loop :for i :from n :downto 0
	  :for j :from 0 :to n
	  :do (incf res (* (aref byte-vec i) (expt 256 j))))
    res))

(defun img-bytes-to-floats (byte-vec)
  "Takes in a byte vector of the image and produces a float vector
   of normalised pixel values"
  (let* ((n (length byte-vec))
	 (result (make-array n :element-type 'single-float)))
    (dotimes (i n result)
      (setf (aref result i)
	    (/ (coerce (aref byte-vec i) 'single-float) 255)))))

(defun print-img (img-buffer)
  (format t "~%")
  (dotimes (i 28)
    (dotimes (j 28)
      (let ((pixel (aref img-buffer (+ (* i 28) j))))
	(if (> pixel 0.5)
	    (format t "  ")
	    (format t "##"))))
    (format t "~%")))

(defun img-buffer->tensor (img-buffer)
  "Wraps img buffer in tensor"
  (make-instance 'tensor :dims '(784 1) :buffer img-buffer))


;; train-images.idx3-ubyte
;; idx3-ubyte has the following format:
;;  - 4 byte magic number to identify file type
;;  - 4 byte nimages
;;  - 4 byte nrows
;;  - 4 byte ncols
;;  - 1 byte image data
(defun read-mnist-imgs (filename nimgs)
  "Returns vector of img tensors (784x1 matrices). Embarassingly hardcoded for the moment"
  (with-open-file (in filename :element-type '(unsigned-byte 8))
    (let* ((temp-buffer (make-array 4 :element-type '(unsigned-byte 8)))
	   (img-byte-buffer (make-array (* 28 28) :element-type '(unsigned-byte 8)))
	   (result (make-array nimgs)))
      ;; magic number
      (read-sequence temp-buffer in)
      ;; number of images
      (read-sequence temp-buffer in)
      ;; number of rows
      (read-sequence temp-buffer in)
      ;; number of cols
      (read-sequence temp-buffer in)
      (dotimes (i nimgs result)
	(read-sequence img-byte-buffer in)
	(setf (aref result i)
	      (img-buffer->tensor
	       (img-bytes-to-floats img-byte-buffer)))))))

;; idx1-ubyte has the following format
;; - 4 bytes magic number
;; - 4 bytes number of labels
;; - 1 byte labels
(defun read-mnist-labels (filename nlabels)
  "Returns vector of label tensors (10x1 matrices). Embarassingly hardcoded for the moment"
  (with-open-file (in filename :element-type '(unsigned-byte 8))
    (let* ((temp-buffer (make-array 4 :element-type '(unsigned-byte 8)))
	   (label-byte)
	   (label-tensor)
	   (result (make-array nlabels)))
      ;; magic number
      (read-sequence temp-buffer in)
      ;; number of labels
      (read-sequence temp-buffer in)
      (dotimes (i nlabels result)
	(setf label-byte (read-byte in))
	(setf label-tensor (zeros '(10 1)))
	(setf (tref label-tensor label-byte 0) 1.0)
	(setf (aref result i) label-tensor)))))

(defun column-matrix-to-int (t1)
  "Takes a one hot encoded col matrix and returns the correspinding integer"
  (let ((buffer (buffer t1))
	(idx))
    (dotimes (i (length buffer))
      (when (= (aref buffer i) 1.0)
	(setf idx i)))
    (assert idx)
    idx))

(defun classify-img (fnn x)
  (let* ((y (feed fnn x))
	 (buffer (buffer y))
	 (max-idx 0))
    (dotimes (i (length buffer) max-idx)
      (when (> (aref buffer i) (aref buffer max-idx))
	(setf max-idx i)))))

(defclass test-report ()
  ((failed :initarg :failed
	   :initform nil
	   :accessor failed
	   :documentation
	   "alist of missclassified data (x . y)")
   (ncorrect :initarg :ncorrect
	     :initform 0
	     :accessor ncorrect
	     :documentation
	     "number of correctly classified data")
   (ndata :initarg :ndata
	  :accessor ndata
	  :documentation
	  "Number of data being tested")))

(defmethod print-object ((r test-report) s)
  (format t "~%___TEST_REPORT___~%")
  (format t "~a / ~a Correct~%" (ncorrect r) (ndata r)))

(defun test-model (fnn x-test y-test)
  (let* ((n (length x-test))
	 (report (make-instance 'test-report :ndata n)))
    (for ((x x-test)
	  (y y-test)
	  (count (range n)))
      (format t "Testing image ~a~%" count)
      (if (= (classify-img fnn x)
	     (column-matrix-to-int y))
	  (incf (ncorrect report))
	  (setf (failed report)
		(cons (cons x y) (failed report)))))
    report))

;;;;;;;;; Training and evaluating model ;;;;;;;;;;;;

;; (defparameter *x-train*
;;   (read-mnist-imgs "./mnist/train-images.idx3-ubyte" 60000))

;; (defparameter *y-train*
;;   (read-mnist-labels "./mnist/train-labels.idx1-ubyte" 60000))

;; (defparameter *x-test*
;;   (read-mnist-imgs "./mnist/t10k-images.idx3-ubyte" 10000))

;; (defparameter *y-test*
;;   (read-mnist-labels "./mnist/t10k-labels.idx1-ubyte" 10000))

;; (defparameter *fnn* (new-fnn-model #'sigmoid 784 25 10))

;; (learn-noisy *fnn* *x-train* *y-train* :rate 0.5)

;; (defparameter *report*
;;   (test-model *fnn* *x-test* *y-test*))




