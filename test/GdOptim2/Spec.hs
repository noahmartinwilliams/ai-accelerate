{-# LANGUAGE FlexibleContexts #-}
module Main where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter
import Data.Array.Accelerate.Matrix
import Data.Random.Normal
import ML.AI.ActFuncs
import ML.AI.ANN
import ML.AI.Block
import ML.AI.ErrFn
import ML.AI.Grad
import ML.AI.Optim
import ML.AI.OptimANN
import ML.AI.Optim.GD
import ML.AI.Types
import Prelude as P
import System.Exit
import System.Random

data AInp = AInp
data AOutp = AOutp

mkSamples :: StdGen -> [(Double, Double, Double)]
mkSamples g = let ns = normals g :: [Double] in intern ns where
    intern :: [Double] -> [(Double, Double, Double)]
    intern [] = undefined
    intern [_] = undefined
    intern (a : b : r) | (a P.> 0.0) P.&& (b P.> 0.0) = (a, b, 1.0) : (intern r)
    intern (a : b : r) = (a, b, 0.0) : (intern r)

mkMats :: [(Double, Double, Double)] -> [(Matrix Double, Matrix Double)]
mkMats [] = []
mkMats ((a, b, c) : r) = (A.fromList (Z:.2:.1) [a, b], A.fromList (Z:.1:.1) [c]) : (mkMats r)

trainANN :: (Optim o Double AOutp AInp, Optim o' Double AOutp One) => OptimANN o o' Double AInp AOutp -> Acc (Matrix Double, Matrix Double) -> OptimANN o o' Double AInp AOutp
trainANN ann sample = do
    let (inp, outp) = A.unlift sample :: (Acc (Matrix Double), Acc (Matrix Double))
    let (prevIns, (AccMat trueOut _ One)) = inferOptimANN (AccMat inp AInp One) ann
        trueOut' = AccMat trueOut AOutp One
        bp = ((derrFn MSEFn ) (AccMat outp AOutp One) trueOut')
        (grads, _) = gradOptimANN ann prevIns bp
        ann' = bpANN ann (grad2block grads AInp)
    ann'
        
trainANN' :: ANNBlInfo -> Acc (Vector Int, Vector Double) -> Acc (Matrix Double, Matrix Double) -> Acc (Vector Int, Vector Double)
trainANN' blinfo optimizer samp = do
    let (AccBlock optimizer' _) = annOptim2block (trainANN (block2annOptim blinfo (AccBlock optimizer (ANNBlock, AInp, AOutp)) :: OptimANN GradDescOptim GradDescOptim Double AInp AOutp) samp)
    optimizer'



type Fn = ((Vector Int, Vector Double) -> (Matrix Double, Matrix Double) -> (Vector Int, Vector Double))

trainer :: Fn -> (Vector Int, Vector Double) -> [(Matrix Double, Matrix Double)] -> (Vector Int, Vector Double)
trainer _ o [] = o
trainer fn o (h : t) = let o' = fn o h in trainer fn o' t

tester :: (Matrix Double -> Matrix Double) -> [(Matrix Double, Matrix Double)] -> Double
tester _ [] = 0.0
tester fn ((hi, ho) : t) = do
    let o = fn hi
        ol = A.toList o
        hol = A.toList ho
    (P.sum (P.zipWith (\x -> \y -> (x - y) * (x - y)) ol hol)) + (tester fn t)

main :: IO ()
main = do
    let g = mkStdGen 100
        g' = mkStdGen 101
        (bli, ann) = mkANN g 2 Sigmoid [(4, Sigmoid), (3, Sigmoid), (1, Sigmoid)] AInp AOutp
        opt = GradDescOptim { gdoLR = constant 0.001} :: GradDescOptim Double AOutp AInp
        opt' = GradDescOptim { gdoLR = constant 0.001 } :: GradDescOptim Double AOutp One
        optAnn = mkOptimANN ann opt  opt'
        (AccBlock optBlock _) = annOptim2block optAnn
        samps = mkSamples g'
        samps' = mkMats samps
        sampsTest = P.take 1024 samps'
        sampsTrain = P.take 1024 (P.drop 1024 samps')
        trainFn = runN (trainANN' bli) 
        firstAttemptFn = runN (\x -> let (_, (AccMat outp _ _)) = inferANN ann (AccMat x AInp One) in outp)
        firstAttempt = tester firstAttemptFn sampsTest
        trained = trainer trainFn (run optBlock) sampsTrain
        (_, trained') = splitOptimANN (block2annOptim bli (AccBlock (use trained) (ANNBlock, AInp, AOutp)) :: OptimANN GradDescOptim GradDescOptim  Double AInp AOutp)
        trained'' = run (let (AccBlock x _) = P.snd (ann2block trained') in x)
        secondAttemptFn = runN (\x -> let (_, (AccMat outp _ _)) = inferANN (block2ann bli (AccBlock (use trained'') (ANNBlock, AInp, AOutp))) (AccMat x AInp One) in outp) 
        secondAttempt = tester secondAttemptFn sampsTest
    if secondAttempt P.>= firstAttempt 
    then do
        putStrLn ("Failed. Got: first attempt: " P.++ (show firstAttempt) P.++ " second attempt: " P.++ (show secondAttempt) )
        exitFailure
    else
        putStrLn ("Success. Got: first attempt: " P.++ (show firstAttempt) P.++ " second attempt: " P.++ (show secondAttempt))
