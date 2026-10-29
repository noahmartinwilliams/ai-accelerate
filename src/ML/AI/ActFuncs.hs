{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
module ML.AI.ActFuncs where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Prelude as P

data ActFunc = Sigmoid | ReLU | SoftMax deriving(Show, P.Eq)

actFunc :: (A.Floating f, P.Fractional f, A.Ord f) => ActFunc -> AccMat f a b -> AccMat f a b
actFunc Sigmoid x = let one = constant 1.0 in matMap x (\w -> one / (one + (exp (-w))))
actFunc ReLU x = matMap x (\y -> A.max (constant 0.0) y)
actFunc SoftMax x = do
    let m y = matMaximum' y
    let (AccMat es a b) = matMap x (\y -> exp (y - (m x)))
        s = A.sum (A.flatten es)
    matMap (AccMat es a b) (\z -> z / (A.the s))


dactFunc :: (A.Floating f, P.Fractional f, A.Ord f) => ActFunc -> AccMat f a b -> AccMat f a b
dactFunc Sigmoid x = do
    let one = constant 1.0
        e y = exp (-y)
        e' y = (e y) + one
    matMap x (\z -> (e z) / ((e' z) * (e' z))) 
dactFunc ReLU x = do
    matMap x (\y -> (y A.> (constant 0.0)) A.? (constant 1.0, constant 0.0))
dactFunc SoftMax x = do
    let s = actFunc SoftMax x
        s' = s `matZipMul` s
    s `matSub` s'
