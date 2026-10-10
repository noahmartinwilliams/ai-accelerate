{-# LANGUAGE GADTs, RankNTypes, ScopedTypeVariables, TypeFamilies, TypeOperators, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.ANN where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Data.Random.Normal
import ML.AI.ActFuncs
import ML.AI.Block
import ML.AI.Layer
import ML.AI.Misc
import ML.AI.Optim
import ML.AI.Types
import Prelude as P
import System.Random

data ANNLayer f a b = ANNLayer { annL :: Layer f a b, annActFunc :: ActFunc }

data ANN f a b where
    ANN :: (A.Fractional f, A.Floating f) => [ANNLayer f a b] -> a -> b -> ANN f a b

data OptimANNLayer o o' f a b = OptimANNLayer { oalLayer :: ANNLayer f a b, oalweights :: o f b a, oalbiases :: o' f b One }

data OptimANN o o' f a b where
    OptimANN :: (A.Floating f) => [OptimANNLayer o o' f a b] -> a -> b -> OptimANN o o' f a b 

data ANNBlock = ANNBlock

data ANNOptimBlock = ANNOptimBlock

data ANNBlInfo = ANNBlInfo [ActFunc] deriving(Show)

mkOptimANN :: (A.Fractional f, Optim o f b a, Optim o' f b One ) => ANN f a b -> o f b a -> o' f b One -> OptimANN o o' f a b
mkOptimANN (ANN [] a b ) _ _ = OptimANN [] a b
mkOptimANN (ANN ( h : t ) a b ) optimizer optimizer' = do
    let (OptimANN rest _ _) = mkOptimANN (ANN t a b) optimizer optimizer'
        matW = optimBuildForMat optimizer (lweights (annL h))
        matB = optimBuildForMat optimizer' (lbiases (annL h))
    OptimANN ((OptimANNLayer { oalLayer = h, oalweights = matW, oalbiases = matB }) : rest) a b
        

splitOptimANN :: (A.Fractional f, Optim o f b a, Optim o' f b One) => OptimANN o o' f a b -> (AccBlock f OptimBlock, ANN f a b)
splitOptimANN (OptimANN layers a b) = do
    let alayers = P.map oalLayer layers
        weights = P.map oalweights layers
        biases = P.map oalbiases layers
        optimizer weights' biases' r = appendAccBlocks (appendAccBlocks (optim2block weights' OptimBlock) (optim2block biases' OptimBlock) OptimBlock ) r OptimBlock
        optimizerBlock = P.foldr (\(ws, bs) -> \r -> optimizer ws bs r) (accEmptyBlock OptimBlock) (P.zip weights biases)
    (optimizerBlock, ANN alayers a b)

    
inferANNLayer :: (P.Fractional f, A.Floating f, A.Ord f) => AccMat f a One -> [ ANNLayer f a b ] -> ([AccMat f a One], AccMat f a One)
inferANNLayer inp [] = ([], inp)
inferANNLayer inp@(AccMat _ inpType _) (h : t) = do
    let x@(AccMat x' _ One) = inferLayer (annL h) inp
        af = annActFunc h
        (AccMat outp _ One) = actFunc af x 
        (restList, (AccMat outp' _ One)) = inferANNLayer (AccMat outp inpType One) t
    ((AccMat x' inpType One) : restList, (AccMat outp' inpType One))

inferANN :: (P.Fractional f, A.Floating f, A.Ord f) => ANN f a b -> AccMat f a One -> ([AccMat f a One], AccMat f a One)
inferANN (ANN ls _ _) inp = inferANNLayer inp ls

inferOptimANN :: (P.Fractional f, A.Ord f) => AccMat f a One -> OptimANN o o' f a b -> ([AccMat f a One], AccMat f b One)
inferOptimANN inp (OptimANN ls a b ) = let ann = ANN (P.map oalLayer ls) a b in
    let (r, AccMat outp _ _ ) = inferANN ann inp in (r, AccMat outp b One)

gradANN :: (P.Fractional f, A.Floating f, A.Ord f) => ANN f a b -> [AccMat f a One] -> AccMat f b One -> ([(AccMat f b a, AccMat f b One)], AccMat f a One)
gradANN (ANN _ a _)  [] (AccMat amat _ _) = ([], AccMat amat a One)
gradANN (ANN [] _ _ ) _ (AccMat _ _ _) = undefined
gradANN (ANN (ah : at) a b) (h : t) bp = do
    let (l, (AccMat bprop _ One)) = gradANN (ANN at a b) t bp
        bprop' = AccMat bprop b One
        af = annActFunc ah
        dbp = dactFunc af bprop'
        (wg, bg, bprop'') = gradLayer (annL ah) h dbp
    ((wg, bg) : l, bprop'')

gradOptimANN :: (P.Fractional f, A.Fractional f, A.Ord f) => OptimANN o o' f a b -> [AccMat f a One] -> AccMat f b One -> ([(AccMat f b a, AccMat f b One)], AccMat f a One)
gradOptimANN (OptimANN ls a b) ls' bp = let ann = (ANN (P.map oalLayer ls) a b) in gradANN ann ls' bp

mkANN :: forall f a b. (A.Floating f, Random f, P.Floating f) => StdGen -> Int -> ActFunc -> [(Int, ActFunc)] -> a -> b -> (ANNBlInfo, ANN f a b)
mkANN g numInputs af list a b = mkANN' g numInputs af ((numInputs, af) : list) a b

mkANN' :: forall f a b. (A.Floating f, Random f, P.Floating f) => StdGen -> Int -> ActFunc -> [(Int, ActFunc)] -> a -> b -> (ANNBlInfo, ANN f a b)
mkANN' _ _ _ [] a b = (ANNBlInfo [], ANN [] a b)
mkANN' g numInputs firstAF ((numOuts, af) : layerSpecRest) a b = do
    let numWeights = numInputs * numOuts
    let rs = normals g :: [f]
        g' = mkStdGen ((randoms g :: [Int]) P.!! 0)
        rs' = heWeightInit numInputs (P.take (numWeights + numOuts) rs)
        rsW = P.take numWeights rs'
        rsB = P.take numOuts (P.drop numWeights rs')
        layer = Layer { lweights = AccMat (use (A.fromList (Z:.numOuts:.numInputs) rsW)) b a, lbiases = AccMat (use (A.fromList (Z:.numOuts:.1) rsB)) b One}
        annLayer = ANNLayer { annL = layer, annActFunc = firstAF }
        (ANNBlInfo restBlInfo, nextLayers) = mkANN' g' numOuts af layerSpecRest a b 
        (ANN nextLayers' _ _) = nextLayers
    (ANNBlInfo (firstAF : restBlInfo),  ANN (annLayer : nextLayers') a b)

heWeightInit :: (P.Floating f) => Int -> [f] -> [f]
heWeightInit numIns fs = let numInsF = P.fromIntegral numIns in P.map  (\x -> x * 2.0 / (sqrt numInsF)) fs

bpANN :: (A.Fractional f, Optim o f b a, Optim o' f b One) => OptimANN o o' f a b -> [(AccMat f b a, AccMat f b One)] -> OptimANN o o' f a b
bpANN o [] = o
bpANN (OptimANN [] _ _) _ = undefined
bpANN (OptimANN (oann : ls) a c) ( (wh, bh) : t) = do
    let w = lweights (annL (oalLayer oann))
        b = lbiases (annL (oalLayer oann))
        wo = oalweights oann
        bo = oalbiases oann
        (wo', w') = optim wo w wh
        (bo', b') = optim bo b bh
        oann' = OptimANNLayer { oalLayer = ANNLayer{ annL = Layer{lweights = w', lbiases = b'}, annActFunc = (annActFunc (oalLayer oann))}, oalweights = wo', oalbiases = bo'}
        (OptimANN optimizer _ _) = bpANN (OptimANN ls a c) t
    OptimANN (oann' : optimizer) a c

block2annLayer :: forall f a b. (A.Fractional f) => ActFunc -> AccBlock f (ANNBlock, a, b) -> (ANNLayer f a b, AccBlock f (ANNBlock, a, b))
block2annLayer af ablock@(AccBlock _ (ANNBlock, a, c)) = do
    let (w, r) = accBlockGetMat ablock c a (ANNBlock, Unk, c)
        (numOuts, _) = accGetFirstIntBlock ablock
        (b, (AccBlock r' _)) = accTakeNFs r numOuts
        b' = A.reshape (A.lift (Z:.numOuts:.(constant 1))) b
    (ANNLayer { annL = Layer { lweights = w, lbiases = AccMat b' c One}, annActFunc = af }, AccBlock r' (ANNBlock, a, c))

block2ann :: (A.Floating f) => ANNBlInfo -> AccBlock f (ANNBlock, a, b) -> ANN f a b 
block2ann (ANNBlInfo []) (AccBlock _ (_, a, b)) = ANN [] a b
block2ann (ANNBlInfo (af : restAF)) ablock@(AccBlock _ (ANNBlock, a, b)) = do
    let (numOutputs, rest) = accGetFirstIntBlock ablock
        (numInputs, rest') = accGetFirstIntBlock rest
        numWeights = numOutputs * numInputs
        (weightsV, postWeights) = accTakeNFs rest' numWeights
        (biasesV, postBiases) = accTakeNFs postWeights numOutputs
        weightsM = A.reshape (A.lift (Z:.numOutputs:.numInputs)) weightsV
        biasesM = A.reshape (A.lift (Z:.numOutputs:.(constant 1))) biasesV
        (ANN restANN _ _ ) = block2ann (ANNBlInfo restAF) postBiases
    (ANN ((ANNLayer { annActFunc = af, annL = Layer { lweights = (AccMat weightsM b a), lbiases = (AccMat biasesM b One)}}) : restANN) a b)

ann2block :: (A.Fractional f) => ANN f a b -> (ANNBlInfo, AccBlock f (ANNBlock, a, b))
ann2block (ANN [] a b ) = (ANNBlInfo [], accEmptyBlock (ANNBlock, a, b))
ann2block (ANN (annLayer : annRest) a b ) = do
    let af = annActFunc annLayer
        alayer = annL annLayer
        weights = lweights alayer
        (AccMat biases _ One) = lbiases alayer
        (vi, vd) = A.unlift (accGetBlock (accMatGetBlock weights Unk))
        biases' = A.flatten biases
        vd' = vd A.++ biases'
        (ANNBlInfo blinfo', rest) = ann2block (ANN annRest a b)
        (ri, rd) = A.unlift (accGetBlock rest)
    (ANNBlInfo (af : blinfo'), AccBlock (A.lift (vi A.++ ri, vd' A.++ rd)) (ANNBlock, a, b))

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
        (_, AccBlock block _) = ann2block (ANN [l] a b)
        block' = annOptim2block (OptimANN t a b)
        typ = (ANNOptimBlock, a, b)
    appendAccBlocks (AccBlock block typ) (appendAccBlocks (appendAccBlocks (optim2block we typ) (optim2block bi typ) typ) block' typ) typ
