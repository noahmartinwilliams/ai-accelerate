{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
module ML.AI.Params where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Layer 
import Prelude as P

data Params f = Params { pweights :: AccMat f Outp Inp, pbiases :: AccMat f Outp One }

instance (P.Fractional f, A.Elt f) => Layer Params f where
    lweights p = pweights p
    lbiases p = pbiases p
    l2Block p@(Params { pweights = w, pbiases = b}) = appendAccBlocks (accMatGetBlock w (Outp, Inp)) (accMatGetBlock b (Outp, One)) p
