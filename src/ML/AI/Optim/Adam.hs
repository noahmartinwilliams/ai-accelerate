{-# LANGUAGE RankNTypes, ScopedTypeVariables, TypeFamilies, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.Optim.Adam where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Layer
import ML.AI.Misc
import ML.AI.Optim
import ML.AI.OptimANN
import ML.AI.Optim.Mom
import ML.AI.Optim.RMS
import ML.AI.Types
import Prelude as P

data AdamOptim f a b = AdamOptim { adamoMomentum :: MomentOptim f a b , adamoRMS :: RMSOptim f a b, adamoBeta1t :: Exp f, adamoBeta2t :: Exp f }

instance forall f a b. (P.Fractional f, A.Floating f) => Optim AdamOptim f a b where
    optimBuildForMat adam@(AdamOptim {adamoMomentum = mom, adamoRMS = rms}) mat = do
        let rms' = optimBuildForMat rms mat
            mom' = optimBuildForMat mom mat
        (adam { adamoMomentum = mom', adamoRMS = rms'})

    optim adam@(AdamOptim { adamoMomentum = mom, adamoRMS = rms, adamoBeta1t = b1t, adamoBeta2t = b2t }) params dw = do
        let beta1t' = b1t * (momoBeta mom)
            beta2t' = b2t * (rmsoGamma rms)
            (mom', _) = optim mom params dw
            (rms', _) = optim rms params dw
            mtHat = matMap (momoVelocity mom') (\x -> x / ((constant 1.0) - beta1t'))
            vtHat = matMap (rmsoGradAvg rms') (\x -> x / ((constant 1.0) - beta2t'))
            gradUpdate = matZipDiv mtHat (matMap vtHat (\x -> (sqrt x) + 0.00000001))
            params' = params `matSub` ((momoAlpha mom) `matScale` gradUpdate)
        (adam { adamoMomentum = mom', adamoRMS = rms', adamoBeta1t = beta1t', adamoBeta2t = beta2t'}, params')

    optim2block (AdamOptim { adamoMomentum = mom, adamoRMS = rms, adamoBeta1t = b1t, adamoBeta2t = b2t }) c = do
        let vi = use (A.fromList (Z:.1) [4] :: Vector Int)
            momBlock = optim2block mom Unk
            rmsBlock  = optim2block rms Unk
            b1tv = rep0 (constant 1) (A.unit b1t)
            b2tv = rep0 (constant 1) (A.unit b2t)
            vd = b1tv A.++ b2tv
            block = AccBlock (A.lift (vi, vd)) Unk
        appendAccBlocks block (appendAccBlocks momBlock rmsBlock c) c

    block2optim (AccBlock bl c) a b d = do
        let (vi, vd) = A.unlift bl
            vi' = A.drop (constant 1) vi
            b1t = vd A.!! (constant 0)
            b2t = vd A.!! (constant 1)
            vd' = A.drop (constant 2) vd
            block = AccBlock (A.lift (vi', vd')) c
            (mom, restBlock) = block2optim block a b d
            mom' = mom :: MomentOptim f a b
            (rms, restBlock') = block2optim restBlock a b d 
            rms' = rms :: RMSOptim f a b
        (AdamOptim { adamoMomentum = mom', adamoRMS = rms', adamoBeta1t = b1t, adamoBeta2t = b2t }, restBlock')
