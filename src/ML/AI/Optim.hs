{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts #-}
module ML.AI.Optim where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Layer
import ML.AI.Params
import Prelude as P

class (  P.Num (Exp f), A.Elt f) => Optim o f a b where
    optim :: o f a b -> AccMat f a b -> AccMat f a b -> (o f a b, AccMat f a b)
