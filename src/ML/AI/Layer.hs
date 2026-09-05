{-# LANGUAGE MultiParamTypeClasses #-}
module ML.AI.Layer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import Prelude as P

data One = One
data Outp = Outp
data Inp = Inp

class (P.Num f, A.Elt f) => Layer l f where
    lweights :: l f -> AccMat f Outp Inp
    lbiases :: l f -> AccMat f Outp One
    lGetBlock :: l f -> AccBlock f (l f)

