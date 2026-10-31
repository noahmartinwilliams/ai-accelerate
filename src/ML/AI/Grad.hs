{-# LANGUAGE RankNTypes, FlexibleContexts, ScopedTypeVariables #-}
module ML.AI.Grad where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.ActFuncs
import ML.AI.ANN
import ML.AI.Block
import ML.AI.Layer
import ML.AI.OptimANN
import ML.AI.Types
import Prelude as P

gradAddDs :: forall f c. (A.Floating f) => AccBlock f c -> AccBlock f c -> AccBlock f c 
gradAddDs (AccBlock block1 c) (AccBlock block2 _) = do
    let (_, block1') = A.unlift block1 :: (Acc (Vector Int), Acc (Vector f))
        (vi, block2') = A.unlift block2 :: (Acc (Vector Int), Acc (Vector f))
        block3 = A.zipWith (+) block1' block2' 
    AccBlock (A.lift (vi, block3) :: Acc (Vector Int, Vector f)) c

gradANN :: forall f a b. (P.Fractional f, A.Floating f, A.Ord f) => Int -> ANN f a b -> [AccMat f a One] -> AccMat f b One -> ([(AccMat f b a, AccMat f b One)], AccMat f a One)
gradANN _ (ANN _ a _)  [] (AccMat amat _ _) = ([], AccMat amat a One)
gradANN _ (ANN [] _ _ ) _ (AccMat _ _ _) = undefined
gradANN bsize (ANN (ah : at) a b) (h : t) bp = do
    let (l, (AccMat bprop _ One)) = gradANN bsize (ANN at a b) t bp
    let bprop' = AccMat bprop b One
        af = annActFunc ah
        x = inferLayer (annL ah) h
        delta = if af P.== SoftMax then softMaxBP x bprop' else (dactFunc af x) `matZipMul` bprop'
        (wg, bg, bprop'') = gradLayer (annL ah) h delta
        bsize' = P.fromIntegral bsize :: f
        wg' = ((constant 1.0) / (constant bsize')) `matScale` wg
        bg' = ((constant 1.0) / (constant bsize')) `matScale` bg
    ((wg', bg') : l, bprop'')

-- Softmax's Jacobian isn't diagonal, so multiplying by its element-wise derivative drops the
-- cross terms. The full Jacobian-vector product is s * (g - sum (s * g)).
softMaxBP :: (P.Fractional f, A.Floating f, A.Ord f) => AccMat f b One -> AccMat f b One -> AccMat f b One
softMaxBP x g = do
    let s = actFunc SoftMax x
        (AccMat sg _ _) = s `matZipMul` g
        dot = A.the (A.sum (A.flatten sg))
    s `matZipMul` (matMap g (\y -> y - dot))

gradOptimANN :: (P.Fractional f, A.Fractional f, A.Ord f) => Int -> OptimANN o o' f a b -> [AccMat f a One] -> AccMat f b One -> ([(AccMat f b a, AccMat f b One)], AccMat f a One)
gradOptimANN bsize (OptimANN ls a b) ls' bp = let ann = (ANN (P.map oalLayer ls) a b) in gradANN bsize ann ls' bp

grad2block :: forall float a b c. (A.Fractional float) => [(AccMat float b a, AccMat float b One)] -> c -> AccBlock float c
grad2block [] c = accEmptyBlock c
grad2block ((w, b) : rest) c = do
    let w' = accMatGetBlock w c
        b' = accMatGetBlock b c
        h = appendAccBlocks w' b' c
        r = grad2block rest c
    appendAccBlocks h r c

block2grad :: forall float a b c . (A.Fractional float) => Int -> AccBlock float c -> a -> b -> [(AccMat float b a, AccMat float b One)]
block2grad 0 _ _ _ = []
block2grad numMats block a b = do
    let (matW, r) = accBlockGetMat block b a Unk
        (matB, r') = accBlockGetMat r b One Unk
        rest = block2grad (numMats - 1) r' a b
    (matW, matB) : rest
