{-# LANGUAGE GADTs, RankNTypes, ScopedTypeVariables, TypeFamilies, TypeOperators, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.Block where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Misc
import ML.AI.Types


type Block f = (Vector Int, Vector f)

data AccBlock f a where
    AccBlock :: (A.Fractional f) => Acc (Block f) -> a -> AccBlock f a

accGetBlock :: AccBlock f a -> Acc (Vector Int, Vector f)
accGetBlock (AccBlock vects _) = vects

appendAccBlocks :: (A.Elt f) => AccBlock f a -> AccBlock f b -> c -> AccBlock f c
appendAccBlocks (AccBlock v1 _) (AccBlock v2 _) c = do
    let (v1i, v1d) = A.unlift v1 
        (v2i, v2d) = A.unlift v2
        vreti = v1i A.++ v2i
        vretd = v1d A.++ v2d
    (AccBlock (A.lift (vreti, vretd)) c)

accMatGetBlock :: forall f a b c. (A.Fractional f) => AccMat f a b -> c -> AccBlock f c
accMatGetBlock (AccMat f _ _) c = do
    let ds = A.flatten f
        sh = A.shape f
        (Z:.m:.n) = A.unlift sh
        is = rep0 (constant 1) (A.unit m)
        is' = rep0 (constant 1) (A.unit n) 
    AccBlock (A.lift (is A.++ is', ds)) c

accGetFirstIntBlock :: forall f a. (A.Fractional f) => AccBlock f a -> (Exp Int, AccBlock f a )
accGetFirstIntBlock (AccBlock inp a) = do
    let (vi, vd) = A.unlift inp :: (Acc (Vector Int), Acc (Vector f))
    ((vi A.!! (constant 0)), AccBlock (A.lift (A.drop (constant 1) vi, vd)) a)

accTakeNFs :: forall f a. (A.Fractional f) => AccBlock f a -> Exp Int -> (Acc (Vector f), AccBlock f a)
accTakeNFs (AccBlock v a) num = do
    let (vi, vd) = A.unlift v :: (Acc (Vector Int), Acc (Vector f))
        vd' = A.take num vd
        vd'' = A.drop num vd
    (vd', AccBlock (A.lift (vi, vd'')) a)

accEmptyBlock :: (A.Fractional f) => a -> AccBlock f a
accEmptyBlock a = do
    let vi = A.fromList (Z:.0) [] :: Vector Int
        vd = A.fromList (Z:.0) []
    AccBlock (A.lift (vi, vd)) a

accBlockGetMat :: (A.Fractional f) => AccBlock f a -> b -> c -> d -> (AccMat f b c, AccBlock f d)
accBlockGetMat (AccBlock f _) b c d = do
    let (vi, vd) = A.unlift f
        m = (vi A.!! (constant 0))
        n = (vi A.!! (constant 1))
        vi' = A.drop (constant 2) vi
        numWeights = m * n
        weightsV = A.take numWeights vd
        vd' = A.drop numWeights vd
        weightsM = A.reshape (A.lift (Z:.m:.n)) weightsV
    (AccMat weightsM b c, AccBlock (A.lift (vi', vd')) d)

