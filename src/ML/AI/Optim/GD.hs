{-# LANGUAGE RankNTypes, ScopedTypeVariables, TypeFamilies, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.Optim.GD where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Layer
import ML.AI.Optim
import Prelude as P

data GradDescOptim f a b = GradDescOptim { gdoLR :: Exp f }

instance forall f a b. ( A.Num f) => Optim GradDescOptim f a b where
    optim gdo@(GradDescOptim { gdoLR = lr}) params dw = do
        let params' = params `matSub` (lr `matScale` dw)
        (gdo, params')

