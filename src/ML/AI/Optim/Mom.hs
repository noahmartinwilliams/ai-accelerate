{-# LANGUAGE RankNTypes, ScopedTypeVariables, TypeFamilies, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.Optim.Mom where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Misc
import ML.AI.Optim
import Prelude as P

data MomentOptim f a b = MomentOptim { momoAlpha :: Exp f, momoBeta :: Exp f, momoVelocity :: AccMat f a b}

instance forall f a b. (P.Fractional f, A.Floating f) => Optim MomentOptim f a b where
    optimBuildForMat mo mat = do
        let (AccMat mat' a b) = mat
            sh = A.shape mat'
            mv' = A.fill sh (constant 0.0 :: Exp f)
        mo { momoVelocity = AccMat mv' a b}

    optim momo@(MomentOptim { momoAlpha = alpha, momoBeta = beta, momoVelocity = vel }) params dw = do
        let vel' = (beta `matScale` vel) `matAdd` (((constant 1.0 :: Exp f) - beta) `matScale` dw)
            params' = params `matSub` (alpha `matScale` vel')
        (momo { momoVelocity = vel' }, params')

    optim2block (MomentOptim { momoAlpha = alpha, momoBeta = beta, momoVelocity = vel}) c = do
        let firstFewFs = (rep0 (constant 1) (A.unit alpha)) A.++ (rep0 (constant 1) (A.unit beta))
            matFs = accMatGetBlock vel c
            firstInt = (rep0 (constant 1) (A.unit (constant 2))) 
        appendAccBlocks (AccBlock (A.lift (firstInt, firstFewFs)) c) matFs c

    block2optim (AccBlock block typ) a b d = do
        let (vi, vd) = A.unlift block :: (Acc (Vector Int), Acc (Vector f))
            vi' = A.drop (constant 1) vi
            alpha = vd A.!! (constant 0)
            beta = vd A.!! (constant 1)
            vd' = A.drop (constant 2) vd
            blk' = AccBlock (A.lift (vi', vd')) typ
            (optim', rest) = accBlockGetMat blk' a b d
        (MomentOptim { momoAlpha = alpha, momoBeta = beta, momoVelocity = optim'}, rest)
