#!/bin/sbcl --script

(ql:quickload :mcclim)
(in-package :clim-user)
(run-frame-top-level (make-application-frame 'test :width 500 :height 500))
