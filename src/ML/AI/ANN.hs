{-# LANGUAGE DataKinds, GADTs, RankNTypes, ScopedTypeVariables, TypeFamilies, TypeOperators, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.ANN where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Data.Random.Normal
import ML.AI.ActFuncs
import ML.AI.Block
import ML.AI.Layer
import ML.AI.Types
import Prelude as P
import System.Random

data ANNLayer f a b = ANNLayer { annL :: Layer f a b, annActFunc :: ActFunc }

data ANN f a b where
    ANN :: (A.Fractional f, A.Floating f) => [ANNLayer f a b] -> a -> b -> ANN f a b

data ANNBlock = ANNBlock

data ANNBlInfo = ANNBlInfo [ActFunc] deriving(Show)

    
inferANNLayer :: (P.Fractional f, A.Floating f, A.Ord f) => AccMat f a One -> [ ANNLayer f a b ] -> ([AccMat f a One], AccMat f a One)
inferANNLayer inp [] = ([], inp)
inferANNLayer inp@(AccMat _ inpType _) (h : t) = do
    let x = inferLayer (annL h) inp
        af = annActFunc h
        (AccMat outp _ One) = actFunc af x 
        (restList, (AccMat outp' _ One)) = inferANNLayer (AccMat outp inpType One) t
    (inp : restList, (AccMat outp' inpType One))

inferANN :: (P.Fractional f, A.Floating f, A.Ord f) => ANN f a b -> AccMat f a One -> ([AccMat f a One], AccMat f a One)
inferANN (ANN ls _ _) inp = inferANNLayer inp ls


mkANN :: forall f a b. (A.Floating f, Random f, P.Floating f) => StdGen -> Int -> ActFunc -> [(Int, ActFunc)] -> a -> b -> (ANNBlInfo, ANN f a b)
mkANN g numInputs af list a b = mkANN' g numInputs list a b

mkANN' :: forall f a b. (A.Floating f, Random f, P.Floating f) => StdGen -> Int -> [(Int, ActFunc)] -> a -> b -> (ANNBlInfo, ANN f a b)
mkANN' _ _ [] a b = (ANNBlInfo [], ANN [] a b)
mkANN' g numInputs ((numOuts, af) : layerSpecRest) a b = do
    let numWeights = numInputs * numOuts
    let rs = normals g :: [f]
        g' = mkStdGen ((randoms g :: [Int]) P.!! 0)
        rs' = heWeightInit (numInputs ) (P.take (numWeights + numOuts) rs)
        rsW = P.take numWeights rs'
        rsB = P.take numOuts (P.drop numWeights rs')
        layer = Layer { lweights = AccMat (use (A.fromList (Z:.numOuts:.numInputs) rsW)) b a, lbiases = AccMat (use (A.fromList (Z:.numOuts:.1) rsB)) b One}
        annLayer = ANNLayer { annL = layer, annActFunc = af }
        (ANNBlInfo restBlInfo, nextLayers) = mkANN' g' numOuts layerSpecRest a b 
        (ANN nextLayers' _ _) = nextLayers
    (ANNBlInfo (af : restBlInfo),  ANN (annLayer : nextLayers') a b)

heWeightInit :: (P.Floating f) => Int -> [f] -> [f]
heWeightInit numIns fs = let numInsF = P.fromIntegral numIns in P.map  (\x -> x * (sqrt (2.0 / numInsF))) fs


block2annLayer :: forall f a b. (A.Fractional f) => ActFunc -> AccBlock f (ANNBlock, a, b) -> (ANNLayer f a b, AccBlock f (ANNBlock, a, b))
block2annLayer af ablock@(AccBlock _ (ANNBlock, a, c)) = do
    let (w, r) = accBlockGetMat ablock c a (ANNBlock, Unk, c)
        (b, r') = accBlockGetMat r c One (ANNBlock, a, c)
    (ANNLayer { annL = Layer { lweights = w, lbiases = b}, annActFunc = af }, r' )

block2ann :: (A.Floating f) => ANNBlInfo -> AccBlock f (ANNBlock, a, b) -> ANN f a b 
block2ann (ANNBlInfo []) (AccBlock _ (_, a, b)) = ANN [] a b
block2ann (ANNBlInfo (af : restAF)) ablock@(AccBlock _ (ANNBlock, a, b)) = do
    let (weights, rest) = accBlockGetMat ablock b a Unk
        (biases, rest') = accBlockGetMat rest b One (ANNBlock, a, b)
        (ANN restANN _ _ ) = block2ann (ANNBlInfo restAF) rest'
    (ANN ((ANNLayer { annActFunc = af, annL = Layer { lweights = weights, lbiases = biases}}) : restANN) a b)

ann2block :: (A.Fractional f) => ANN f a b -> (ANNBlInfo, AccBlock f (ANNBlock, a, b))
ann2block (ANN [] a b ) = (ANNBlInfo [], accEmptyBlock (ANNBlock, a, b))
ann2block (ANN (annLayer : annRest) a b ) = do
    let af = annActFunc annLayer
        alayer = annL annLayer
        weights = lweights alayer
        biases = lbiases alayer
        weights' = accMatGetBlock weights (ANNBlock, a, b)
        biases' = accMatGetBlock biases (ANNBlock, b, One)
        (ANNBlInfo blinfo', rest) = ann2block (ANN annRest a b)
        ret = appendAccBlocks weights' biases' (ANNBlock, a, b)
        ret' = appendAccBlocks ret rest (ANNBlock, a, b)
    (ANNBlInfo (af : blinfo'), ret')

