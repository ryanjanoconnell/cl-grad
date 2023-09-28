(in-package :cl-grad)

;; useful functions and macros for working with datasets

(defun shuffle (vec)
  "Fisher yates shuffle"
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

(defun map-shuffle (fn seq1 &rest seqs)
  (let ((len (length seq1)))
    (assert (and (every (lambda (seq) (equal len (length seq))) seqs)))
    (let ((ids (shuffle (range len))))
      (loop :for idx :across ids :do
	(apply fn
	       (elt seq1 idx)
	       (mapcar (lambda (seq) (elt seq idx)) seqs))))))

(defmacro for-shuffle (binds &body body)
  (let ((syms (mapcar #'first binds))
	(seqs (mapcar #'second binds)))
    `(map-shuffle (lambda ,syms ,@body) ,@seqs)))

(defmacro for (binds &body body)
  (let ((syms (mapcar #'first binds))
	(seqs (mapcar #'second binds)))
    `(map nil (lambda ,syms ,@body) ,@seqs)))




