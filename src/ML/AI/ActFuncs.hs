{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
module ML.AI.ActFuncs where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Prelude as P

data ActFunc = Sigmoid | ReLU deriving(Show, P.Eq)

actFunc :: (A.Floating f, P.Fractional f, A.Ord f) => ActFunc -> AccMat f a b -> AccMat f a b
actFunc Sigmoid x = let one = constant 1.0 in matMap x (\w -> one / (one + (exp (-w))))
actFunc ReLU x = matMap x (\y -> A.max (constant 0.0) y)


dactFunc :: (A.Floating f, P.Fractional f, A.Ord f) => ActFunc -> AccMat f a b -> AccMat f a b
dactFunc Sigmoid x = do
    let one = constant 1.0
        e y = exp (-y)
        e' y = (e y) + one
    matMap x (\z -> (e z) / ((e' z) * (e' z))) 
dactFunc ReLU x = do
    matMap x (\y -> (y A.> (constant 0.0)) A.? (constant 1.0, constant 0.0))
