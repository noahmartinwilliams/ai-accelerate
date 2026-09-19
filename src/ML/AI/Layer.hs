{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts #-}
module ML.AI.Layer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import Prelude as P

data One = One deriving(P.Eq)
data Outp = Outp deriving(P.Eq)
data Inp = Inp deriving(P.Eq)

class (A.Fractional f) => Layer l f a b where
    lweights :: l f a b -> AccMat f b a
    lbiases :: l f a b -> AccMat f b One
    l2block :: l f a b -> AccBlock f (l f a b)
    block2l :: AccBlock f (l f a b ) -> a -> b -> (l f a b, AccBlock f (l f a b))


inferLayer :: (Layer l f a b ) => l f a b -> AccMat f a One -> AccMat f b One
inferLayer p inp = do
    let w = lweights p
        biases = lbiases p 
    (w `matMul` inp) `matAdd` biases
