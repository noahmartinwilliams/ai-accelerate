module Main where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Data.Array.Accelerate.Interpreter
import Data.Random.Normal
import ML.AI.ActFuncs
import ML.AI.ANN
import ML.AI.Block
import ML.AI.ErrFn
import ML.AI.Optim
import ML.AI.Optim.Mom
import ML.AI.Types
import Prelude as P
import System.Exit
import System.Random

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

type Fn = ((Vector Int, Vector Double) -> (Matrix Double, Matrix Double) -> (Vector Int, Vector Double))

trainer :: Fn -> (Vector Int, Vector Double) -> [(Matrix Double, Matrix Double)] -> (Vector Int, Vector Double)
trainer _ o [] = o
trainer fn o (h : t) = let o' = fn o h in trainer fn o' t

trainer' :: ANNBlInfo -> Acc (Vector Int, Vector Double) -> Acc (Matrix Double, Matrix Double) -> Acc (Vector Int, Vector Double)
trainer' bli ann samp = do
    let (inp, outp) = A.unlift samp :: (Acc (Matrix Double), Acc (Matrix Double))
        annTyped = block2annOptim bli (AccBlock ann (ANNBlock, Inp, Outp)) :: OptimANN MomentOptim  MomentOptim Double Inp Outp
        (prevIns, (AccMat trueOut _ One)) = inferOptimANN (AccMat inp Inp One) annTyped
        trueOut' = AccMat trueOut Outp One
        bp = derrFn MSEFn (AccMat outp Outp One) trueOut'
        (grads, _) = gradOptimANN annTyped prevIns bp
        ann' = bpANN annTyped grads
        (AccBlock ann'' _) = annOptim2block ann'
    ann''

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
        opt = MomentOptim { momoAlpha = constant 0.01, momoBeta = constant 0.9, momoVelocity = AccMat (use (A.fromList (Z:.1:.1) [0.0])) Outp Inp }
        opt2 = MomentOptim { momoAlpha = constant 0.01, momoBeta = constant 0.9, momoVelocity = AccMat (use (A.fromList (Z:.1:.1) [0.0])) Outp One }
        (bli, ann) = mkANN g 2 ReLU [(4, ReLU), (3, ReLU), (1, Sigmoid)] Inp Outp
        optANN = mkOptimANN ann opt opt2
        (AccBlock optBlock _) = annOptim2block optANN
        samps = mkSamples g
        samps' = mkMats samps
        trainFn = runN (trainer' bli)
        sampsTest = P.take 1024 samps'
        sampsTrain = P.take 1024 (P.drop 1024 samps')
        firstAttemptFn = runN (\x -> let (_, (AccMat outp _ _)) = inferANN ann (AccMat x Inp One) in outp)
        firstAttempt = tester firstAttemptFn sampsTest
        trained = trainer trainFn (run optBlock) sampsTrain
        (_, trained') = splitOptimANN (block2annOptim bli (AccBlock (use trained) (ANNBlock, Inp, Outp)) :: OptimANN MomentOptim  MomentOptim Double Inp Outp)
        trained'' = run (let (AccBlock x _) = P.snd (ann2block trained') in x)
        secondAttemptFn = runN (\x -> let (_, (AccMat outp _ _)) = inferANN (block2ann bli (AccBlock (use trained'') (ANNBlock, Inp, Outp))) (AccMat x Inp One) in outp) 
        secondAttempt = tester secondAttemptFn sampsTest
    if secondAttempt P.>= firstAttempt 
    then do
        putStrLn ("Failed. Got: first attempt: " P.++ (show firstAttempt) P.++ " second attempt: " P.++ (show secondAttempt) )
        exitFailure
    else
        putStrLn ("Success. Got: first attempt: " P.++ (show firstAttempt) P.++ " second attempt: " P.++ (show secondAttempt))
