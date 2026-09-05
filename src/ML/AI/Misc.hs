{-# LANGUAGE RankNTypes, TypeOperators, TypeApplications, ScopedTypeVariables #-}
module ML.AI.Misc where

import Data.Array.Accelerate as A
import Prelude as P


-- This line of code stolen from here: https://hackage-content.haskell.org/package/accelerate-1.4.0.0/docs/Data-Array-Accelerate.html
rep0 :: forall sh e. (Shape sh, Elt e)  => Exp Int -> Acc (Array sh e) -> Acc (Array (sh :. Int) e)
rep0 n a = A.replicate (A.lift (Any @sh:.n)) a
