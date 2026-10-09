{-# LANGUAGE RankNTypes, ScopedTypeVariables, TypeFamilies, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.Optim.GD where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Layer
import ML.AI.Misc
import ML.AI.Optim
import ML.AI.Types

data GradDescOptim f a b = GradDescOptim { gdoLR :: Exp f }


instance forall f a b. ( A.Floating f) => Optim GradDescOptim f a b where
    optimBuildForMat gdo _ = gdo

    optim gdo@(GradDescOptim { gdoLR = lr}) params dw = do
        let params' = params `matSub` (lr `matScale` dw)
        (gdo, params')

    optim2block (GradDescOptim { gdoLR = lr }) a = do
        let vi = use (A.fromList (Z:.0) [] :: Vector Int)
            vd = rep0 (constant 1) (A.unit lr)
        AccBlock (A.lift (vi, vd)) a

    block2optim (AccBlock block _) d = do
        let (vi, vd) = A.unlift block :: (Acc (Vector Int), Acc (Vector f))
            lr = vd A.!! (constant 0)
            vd' = A.drop (constant 1) vd
        (GradDescOptim { gdoLR = lr }, AccBlock (A.lift (vi, vd')) d)

{-
gdOptim :: forall f. (A.Num f) => Acc (Vector Int, Vector f) -> Acc (Matrix f) -> Acc (Matrix f) -> Acc ((Vector Int, Vector f), (Matrix f))
gdOptim optimizer params grads = do
    let optimizer' = block2optim optimizer :: GradDescOptim f Unk Unk
        (optimizer'', (AccMat params' _ _ )) = optim optimizer' (AccMat params Unk Unk) (AccMat grads Unk Unk)
    A.lift (optim2block optimizer'', params')
-}
