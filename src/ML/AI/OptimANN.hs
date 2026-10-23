{-# LANGUAGE RankNTypes, FlexibleContexts, GADTs #-}
module ML.AI.OptimANN where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.AI.ANN
import ML.AI.Block
import {-# SOURCE #-} ML.AI.Grad
import ML.AI.Layer
import ML.AI.Optim
import ML.AI.Types
import Prelude as P

data OptimANNLayer o o' f a b = OptimANNLayer { oalLayer :: ANNLayer f a b, oalweights :: o f b a, oalbiases :: o' f b One }

data OptimANN o o' f a b where
    OptimANN :: (A.Floating f) => [OptimANNLayer o o' f a b] -> a -> b -> OptimANN o o' f a b 

data ANNOptimBlock = ANNOptimBlock

mkOptimANN :: (A.Fractional f, Optim o f b a, Optim o' f b One ) => ANN f a b -> o f b a -> o' f b One -> OptimANN o o' f a b
mkOptimANN (ANN [] a b ) _ _ = OptimANN [] a b
mkOptimANN (ANN ( h : t ) a b ) optimizer optimizer' = do
    let (OptimANN rest _ _) = mkOptimANN (ANN t a b) optimizer optimizer'
        matW = optimBuildForMat optimizer (lweights (annL h))
        matB = optimBuildForMat optimizer' (lbiases (annL h))
    OptimANN ((OptimANNLayer { oalLayer = h, oalweights = matW, oalbiases = matB }) : rest) a b

inferOptimANN :: (P.Fractional f, A.Ord f) => AccMat f a One -> OptimANN o o' f a b -> ([AccMat f a One], AccMat f b One)
inferOptimANN inp (OptimANN ls a b ) = let ann = ANN (P.map oalLayer ls) a b in
    let (r, AccMat outp _ _ ) = inferANN ann inp in (r, AccMat outp b One)

splitOptimANN :: (A.Fractional f, Optim o f b a, Optim o' f b One) => OptimANN o o' f a b -> (AccBlock f OptimBlock, ANN f a b)
splitOptimANN (OptimANN layers a b) = do
    let alayers = P.map oalLayer layers
        weights = P.map oalweights layers
        biases = P.map oalbiases layers
        optimizer weights' biases' r = appendAccBlocks (appendAccBlocks (optim2block weights' OptimBlock) (optim2block biases' OptimBlock) OptimBlock ) r OptimBlock
        optimizerBlock = P.foldr (\(ws, bs) -> \r -> optimizer ws bs r) (accEmptyBlock OptimBlock) (P.zip weights biases)
    (optimizerBlock, ANN alayers a b)

block2annOptim :: forall f o o' a b. (Optim o f b a, Optim o' f b One, A.Floating f) => ANNBlInfo -> AccBlock f (ANNBlock, a, b) -> OptimANN o o' f a b
block2annOptim (ANNBlInfo []) (AccBlock _ (ANNBlock, a, b)) = OptimANN [] a b
block2annOptim (ANNBlInfo (blinfo : rest)) ablock@(AccBlock _ (ANNBlock, a, b)) = do
    let (l, r) = block2annLayer blinfo ablock
        (wo, r') = block2optim r b a (ANNBlock, a, b)
        (bo, r'') = block2optim r' b One (ANNBlock, a, b)
        (OptimANN lrest _ _) = block2annOptim (ANNBlInfo rest) r'' 
    (OptimANN ((OptimANNLayer { oalLayer = l, oalweights = wo, oalbiases = bo}) : lrest) a b)

annOptim2block :: forall f o o' a b. (Optim o f b a, Optim o' f b One, A.Floating f) => OptimANN o o' f a b -> AccBlock f (ANNOptimBlock, a, b) 
annOptim2block (OptimANN [] a b ) = accEmptyBlock (ANNOptimBlock, a, b)
annOptim2block (OptimANN (h : t) a b ) = do 
    let (OptimANNLayer { oalLayer = l, oalweights = we, oalbiases = bi }) = h
        (_, block) = ann2block (ANN [l] a b)
        block' = annOptim2block (OptimANN t a b)
        typ = (ANNOptimBlock, a, b)
    (appendAccBlocks (appendAccBlocks block (appendAccBlocks (optim2block we typ) (optim2block bi typ) typ) typ) block' typ) 

bpANN :: (A.Fractional f, Optim o f b a, Optim o' f b One) => OptimANN o o' f a b -> AccBlock f a -> OptimANN o o' f a b
bpANN (OptimANN ls a c) accblock = do
    let grads = block2grad (P.length ls) accblock a c
        layers = P.map oalLayer ls
        layersL = P.map annL layers
        w = P.zipWith3 optim (P.map (oalweights) ls) (P.map lweights layersL) (P.map P.fst grads)
        b = P.zipWith3 optim (P.map (oalbiases) ls) (P.map lbiases layersL) (P.map P.snd grads)
        b' = P.map P.snd b
        w' = P.map P.snd w
        layers' = P.zipWith3 (\x -> \y -> \z -> x { annL = Layer { lweights = y, lbiases = z}}) layers (w') (b')
        optims = P.zipWith3 (\x -> \y -> \z -> OptimANNLayer { oalLayer = x, oalweights = y, oalbiases = z}) layers' (P.map P.fst w) (P.map P.fst b)
    OptimANN optims a c
