{-# LANGUAGE DataKinds, GADTs, RankNTypes, ScopedTypeVariables, TypeFamilies, TypeOperators, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module ML.AI.ANN where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Data.Random.Normal
import ML.AI.ActFuncs
import ML.AI.Block
import ML.AI.Layer
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

mkANN :: forall f a b. (A.Floating f, Random f, P.Floating f) => StdGen -> Int -> ActFunc -> [(Int, ActFunc)] -> a -> b -> (ANNBlInfo, ANN f a b)
mkANN g numInputs af list a b = mkANN' g numInputs ((numInputs, af) : list) a b

mkANN' :: forall f a b. (A.Floating f, Random f, P.Floating f) => StdGen -> Int -> [(Int, ActFunc)] -> a -> b -> (ANNBlInfo, ANN f a b)
mkANN' _ _ [] a b = (ANNBlInfo [], ANN [] a b)
mkANN' g numInputs ((numOuts, af) : layerSpecRest) a b = do
    let numWeights = numInputs * numOuts
    let rs = normals g :: [f]
        g' = mkStdGen ((randoms g :: [Int]) P.!! 0)
        rs' = heWeightInit numInputs (P.take (numWeights + numOuts) rs)
        rsW = P.take numWeights rs'
        rsB = P.take numOuts (P.drop numWeights rs')
        layer = Layer { lweights = AccMat (use (A.fromList (Z:.numOuts:.numInputs) rsW)) b a, lbiases = AccMat (use (A.fromList (Z:.numOuts:.1) rsB)) b One}
        annLayer = ANNLayer { annL = layer, annActFunc = af }
        (ANNBlInfo restBlInfo, nextLayers) = mkANN' g' numOuts layerSpecRest a b 
        (ANN nextLayers' _ _) = nextLayers
    (ANNBlInfo (af : restBlInfo),  ANN (annLayer : nextLayers') a b)

heWeightInit :: (P.Floating f) => Int -> [f] -> [f]
heWeightInit numIns fs = let numInsF = P.fromIntegral numIns in P.map  (\x -> x * 2.0 / (sqrt numInsF)) fs

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

gradAddDs :: forall f c. (A.Floating f) => AccBlock f c -> AccBlock f c -> AccBlock f c 
gradAddDs (AccBlock block1 c) (AccBlock block2 _) = do
    let (vi, block1') = A.unlift block1 :: (Acc (Vector Int), Acc (Vector f))
        (_, block2') = A.unlift block2 :: (Acc (Vector Int), Acc (Vector f))
        block3 = A.zipWith (+) block1' block2' 
    AccBlock (A.lift (vi, block3) :: Acc (Vector Int, Vector f)) c
