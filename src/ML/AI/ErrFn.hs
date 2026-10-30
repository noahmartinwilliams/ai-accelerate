{-# LANGUAGE ScopedTypeVariables, RankNTypes, FlexibleContexts #-}
module ML.AI.ErrFn where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Prelude as P

data ErrFn = MSEFn | LogEFn deriving(Show, P.Eq)

errFn :: forall f a b . (P.Fractional f, A.Floating f) => ErrFn -> (AccMat f a b -> AccMat f a b -> AccMat f a b )
errFn MSEFn = do
    let squared x = x `matZipMul` x
        half = constant 0.5 
        fn x y = matMap (squared (x `matSub` y)) (\z -> z * half)
    fn
errFn LogEFn = do
    let f x y = matZipWith (\answer -> \actual -> -(answer * (log (actual + epsilon))  )) x y 
    f

derrFn :: forall f a b. (A.Floating f, P.Fractional f) => ErrFn -> (AccMat f a b -> AccMat f a b -> AccMat f a b)
derrFn MSEFn = do
    let fn x y = matMap (x `matSub` y) (\z -> - z )
    fn
derrFn LogEFn = do
    let fn m n = matZipWith (\x -> \y -> -x / (y + epsilon)) m n
    fn 

epsilon :: (A.Floating f, P.Fractional f) => Exp f
epsilon = constant 0.00000000000001
