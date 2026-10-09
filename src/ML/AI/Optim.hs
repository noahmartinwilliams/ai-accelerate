{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts #-}
module ML.AI.Optim where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Layer
import ML.AI.Types
import Prelude as P

data OptimType f a b = OTGD (Exp f) deriving(Show, P.Eq)

class (  P.Num (Exp f), A.Elt f) => Optim o f a b where
    optimBuildForMat :: o f a b -> AccMat f a b -> o f a b
    optim :: o f a b -> AccMat f a b -> AccMat f a b -> (o f a b, AccMat f a b)
    optim2block :: o f a b -> c -> AccBlock f c
    block2optim :: AccBlock f c -> d -> (o f a b , AccBlock f d)

data LayerOptim o f a b = LayerOptim { loweights :: o f b a, lobiases :: o f a One }

data OptimBlock = OptimBlock
