{-# LANGUAGE GADTs #-}
module ML.AI.Block where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Misc
import Prelude as P

data Block f a where
    Block :: P.Fractional f => (Vector Int, Vector f) -> a -> Block f a 

data AccBlock f a where
    AccBlock :: P.Fractional f => Acc (Vector Int, Vector f) -> a -> AccBlock f a

getBlock :: Block f a -> (Vector Int, Vector f)
getBlock (Block vects _) = vects

accGetBlock :: AccBlock f a -> Acc (Vector Int, Vector f)
accGetBlock (AccBlock vects _) = vects

appendAccBlocks :: (P.Fractional f, A.Elt f) => AccBlock f a -> AccBlock f b -> c -> AccBlock f c
appendAccBlocks (AccBlock v1 _) (AccBlock v2 _) c = do
    let (v1i, v1d) = A.unlift v1 
        (v2i, v2d) = A.unlift v2
        vreti = v1i A.++ v2i
        vretd = v1d A.++ v2d
    (AccBlock (A.lift (vreti, vretd)) c)

accMatGetBlock :: P.Fractional f => AccMat f a b -> c -> AccBlock f c
accMatGetBlock (AccMat f _ _) c = do
    let ds = A.flatten f
        sh = A.shape f
        (Z:.m:.n) = A.unlift sh
        is = rep0 (constant 1) (A.unit m)
        is' = rep0 (constant 1) (A.unit n) 
    AccBlock (A.lift (is A.++ is', ds)) c

