{-# LANGUAGE RankNTypes, FlexibleContexts #-}
module ML.AI.ErrFn where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Prelude as P

data ErrFn = MSEFn deriving(Show, P.Eq)

errFn :: forall f a b . (P.Fractional f, A.Num f ) => ErrFn -> (AccMat f a b -> AccMat f a b -> AccMat f a b )
errFn MSEFn = do
    let squared x = x `matZipMul` x
        half = constant 0.5 
        fn x y = matMap (squared (x `matSub` y)) (\z -> z * half)
    fn

derrFn :: forall f a b. (P.Fractional f, A.Num f) => ErrFn -> (AccMat f a b -> AccMat f a b -> AccMat f a b)
derrFn MSEFn = do
    let fn x y = matMap (x `matSub` y) (\z -> - z )
    fn
