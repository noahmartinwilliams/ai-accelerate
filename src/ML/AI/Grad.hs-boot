{-# LANGUAGE RankNTypes, FlexibleContexts, ScopedTypeVariables #-}
module ML.AI.Grad where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.Block
import ML.AI.Types

block2grad :: forall float a b c . (A.Fractional float) => Int -> AccBlock float c -> a -> b -> [(AccMat float b a, AccMat float b One)]
