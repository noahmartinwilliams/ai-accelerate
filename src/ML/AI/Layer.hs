{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts #-}
module ML.AI.Layer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Misc
import Prelude as P

data One = One deriving(P.Eq)
data Outp = Outp deriving(P.Eq)
data Inp = Inp deriving(P.Eq)

data Layer f a b = Layer { lweights :: AccMat f b a, lbiases :: AccMat f b One }

l2block :: (P.Fractional f, A.Fractional f) => Layer f a b -> AccBlock f (Layer f a b)
l2block l@(Layer { lweights = w, lbiases = b}) = appendAccBlocks (accMatGetBlock w (Outp, Inp)) (accMatGetBlock b (Outp, One)) l

l2block' :: (P.Fractional f, A.Fractional f) => Layer f a b -> Acc (Vector Int, Vector f)
l2block' l = do
    let (AccBlock ret _ ) = l2block l
    ret
    

block2l' :: (A.Rational f) => Acc (Vector Int, Vector f) -> a -> b -> (Layer f a b, Acc (Vector Int, Vector f))
block2l' block a b = do
    let (vi, vd) = A.unlift block 
        numOutps = vi A.!! (constant 0)
        numInps = vi A.!! (constant 1)
        vi' = A.drop (constant 2) vi
        numWeights = numOutps * numInps
        weightsV = A.take numWeights vd
        weights = A.reshape (A.lift (Z:.numOutps:.numInps)) weightsV
        biasesV = A.take numOutps (A.drop numWeights vd)
        biases = rep0 (constant 1) biasesV
        rest = A.drop (numWeights + numOutps) vd
        params = Layer { lweights = AccMat weights b a , lbiases = AccMat biases b One}
    (params, (A.lift (vi', rest)))

block2l :: (A.Rational f) => AccBlock f (Layer f a b ) -> a -> b -> (Layer f a b, AccBlock f (Layer f a b))
block2l (AccBlock block c) a b = do
    let (params, ret) = block2l' block a b
    (params, (AccBlock ret c))

inferLayer :: (A.Fractional f) => Layer f a b -> AccMat f a One -> AccMat f b One
inferLayer p inp = do
    let w = lweights p
        biases = lbiases p 
    (w `matMul` inp) `matAdd` biases
