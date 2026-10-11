{-# LANGUAGE RankNTypes, ScopedTypeVariables, TypeFamilies, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.Optim.RMS where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import ML.AI.Block
import ML.AI.Misc
import ML.AI.Optim
import ML.AI.Types
import Prelude as P

data RMSOptim f a b = RMSOptim { rmsAlpha :: Exp f, rmsGamma :: Exp f, rmsGradAvg :: AccMat f a b} 

instance forall f a b. (P.Fractional f, A.Floating f) => Optim RMSOptim f a b where
    optimBuildForMat rmsOptim (AccMat f a b) = do
        let sh = A.shape f
            mat' = A.fill sh (constant 0.0) 
        rmsOptim { rmsGradAvg = AccMat mat' a b}

    optim rmsOptim@(RMSOptim { rmsAlpha = alpha, rmsGamma = gamma, rmsGradAvg = avgGrad }) params dw = do
        let avgGrad' = (avgGrad `matAdd` (gamma `matScale` avgGrad) ) `matAdd` (((constant 1.0) - gamma) `matScale` (dw `matZipMul` dw))
            alphaed = matMap avgGrad (\x -> alpha / (sqrt x))
            params' = params `matSub` (alphaed `matZipMul` dw)
        (rmsOptim { rmsGradAvg = avgGrad' }, params')

    optim2block (RMSOptim { rmsAlpha = alpha, rmsGamma = gamma, rmsGradAvg = gradAvg }) c = do
        let bl = accMatGetBlock gradAvg c
            one = constant 1
            vd = (rep0 one (A.unit alpha)) A.++ (rep0 one (A.unit gamma))
            vi = use (A.fromList (Z:.1) [3] :: Vector Int)
            l = A.lift (vi, vd) :: Acc (Vector Int, Vector f)
            ret = appendAccBlocks (AccBlock l Unk) bl c
        ret

    block2optim (AccBlock block _) a b d = do
        let (vi, vd) = A.unlift block :: (Acc (Vector Int), Acc (Vector f))
            vi' = A.drop (constant 1) vi
            alpha = vd A.!! (constant 0)
            gamma = vd A.!! (constant 1)
            vd' = A.drop (constant 2) vd
            block' = A.lift (vi', vd')
            (m, block'') = accBlockGetMat (AccBlock block' Unk) a b d
        (RMSOptim { rmsAlpha = alpha, rmsGamma = gamma, rmsGradAvg = m }, block'')
