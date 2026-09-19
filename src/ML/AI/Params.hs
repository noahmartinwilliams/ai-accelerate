{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.Params where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Layer 
import ML.AI.Misc
import Prelude as P

data Params f a b = Params { pweights :: AccMat f b a, pbiases :: AccMat f b One }

instance (A.Fractional f, P.Fractional f) => Layer Params f a b where
    lweights p = pweights p
    lbiases p = pbiases p
    l2block p@(Params { pweights = w, pbiases = b}) = appendAccBlocks (accMatGetBlock w (Outp, Inp)) (accMatGetBlock b (Outp, One)) p
    block2l (AccBlock block c) a b = do
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
            params = Params { pweights = AccMat weights b a , pbiases = AccMat biases b One}
        (params, AccBlock (A.lift (vi', rest)) c)


