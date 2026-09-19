{-# LANGUAGE RankNTypes, ScopedTypeVariables, TypeFamilies, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.Optim.GD where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Layer
import ML.AI.Misc
import ML.AI.Optim
import Prelude as P

data GradDescOptim f a b = GradDescOptim { gdoLR :: Exp f }

data Unk = Unk

instance forall f a b. ( A.Num f) => Optim GradDescOptim f a b where
    optim gdo@(GradDescOptim { gdoLR = lr}) params dw = do
        let params' = params `matSub` (lr `matScale` dw)
        (gdo, params')

    optim2block (GradDescOptim { gdoLR = lr }) = do
        let vi = use (A.fromList (Z:.0) [] :: Vector Int)
            vd = rep0 (constant 1) (A.unit lr)
        A.lift (vi, vd)

    block2optim block = do
        let (_, vd) = A.unlift block :: (Acc (Vector Int), Acc (Vector f))
            lr = vd A.!! (constant 0)
        GradDescOptim { gdoLR = lr }

gdOptim :: forall f. (A.Num f) => Acc (Vector Int, Vector f) -> Acc (Matrix f) -> Acc (Matrix f) -> Acc ((Vector Int, Vector f), (Matrix f))
gdOptim optimizer params grads = do
    let optimizer' = block2optim optimizer :: GradDescOptim f Unk Unk
        (optimizer'', (AccMat params' _ _ )) = optim optimizer' (AccMat params Unk Unk) (AccMat grads Unk Unk)
    A.lift (optim2block optimizer'', params')

