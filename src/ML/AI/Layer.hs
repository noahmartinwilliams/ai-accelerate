{-# LANGUAGE MultiParamTypeClasses #-}
module ML.AI.Layer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import Prelude as P

data One = One deriving(P.Eq)
data Outp = Outp deriving(P.Eq)
data Inp = Inp deriving(P.Eq)

class (P.Fractional f, A.Elt f) => Layer l f where
    lweights :: l f -> AccMat f Outp Inp
    lbiases :: l f -> AccMat f Outp One
    l2Block :: l f -> AccBlock f (l f)
