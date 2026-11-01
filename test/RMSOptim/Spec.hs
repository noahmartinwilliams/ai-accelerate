module Main where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.Native
import Data.Array.Accelerate.Matrix
import List.Shuffle
import ML.AI.ActFuncs
import ML.AI.ANN
import ML.AI.Block
import ML.AI.ErrFn
import ML.AI.Grad
import ML.AI.OptimANN
import ML.AI.Optim.RMS
import ML.AI.Types
import Prelude as P
import System.Exit
import System.Random
import Text.Megaparsec
import Text.Megaparsec.CSV

tester :: (Matrix Double -> Matrix Double) -> [(Matrix Double, Matrix Double)] -> Double
tester _ [] = 0.0
tester fn ((hi, ho) : t) = do
    let o = fn hi
        ol = A.toList o
        hol = A.toList ho
    (P.sum (P.zipWith (\x -> \y -> (x - y) * (x - y)) ol hol)) + (tester fn t)

mkSamps :: [[Double]] -> [(Matrix Double, Matrix Double)]
mkSamps ls = do
    let starts = P.map (\x -> P.take 1 x) ls
        rests = P.map (\x -> P.drop 1 x) ls
        startsM = P.map (\x -> A.fromList (Z:.1:.1) x) starts
        restsM = P.map (\x -> A.fromList (Z:.15:.1) x) rests
    P.zip restsM startsM

trainer :: ANNBlInfo -> Acc (Vector Int, Vector Double) -> Acc (Matrix Double, Matrix Double) -> Acc (Scalar Double, (Vector Int, Vector Double))
trainer bli optimANN sampleInp = do
    let (inp, outp) = A.unlift sampleInp :: (Acc (Matrix Double), Acc (Matrix Double))
        optimANN' = block2annOptim bli (AccBlock optimANN (ANNBlock, Inp, Outp)) :: OptimANN RMSOptim RMSOptim Double Inp Outp
        (grads, realOutp) = inferOptimANN (AccMat inp Inp One) optimANN'
        err = (derrFn MSEFn ) (AccMat outp Outp One) realOutp
        (AccMat err' _ _) = (errFn MSEFn) (AccMat outp Outp One) realOutp
        (gradiented, _) = gradOptimANN 1 optimANN' grads err
        bped = bpANN optimANN' (grad2block gradiented Inp)
        (AccBlock ret _) = annOptim2block bped
    A.lift (A.sum (A.flatten err'), ret)

trainer' :: ((Vector Int, Vector Double) -> (Matrix Double, Matrix Double) -> (Scalar Double, (Vector Int, Vector Double))) -> (Vector Int, Vector Double) -> [(Matrix Double, Matrix Double)] -> IO (Vector Int, Vector Double)
trainer' _ f [] = return f
trainer' fn b (h : t) = do
    let (err, ret) = fn b h 
    putStrLn ("error: " P.++ (show ((A.toList err) P.!! 0)))
    trainer' fn ret t

main :: IO ()
main = do
    let fname = "data/food-calories/weight_loss.csv"
    dat <- readFile fname
    let g = mkStdGen 100
        (bli, ann) = mkANN g 15 ReLU [(32, ReLU), (1, ReLU)] Inp Outp
        parsed = runParser (csv ',' Nothing) fname dat
    case parsed of
        (Left str) -> do
            putStrLn (errorBundlePretty str )
            exitFailure
        (Right parsed') -> do
            let parsed'' = P.map (\x -> P.map (\y -> read y :: Double) x) parsed'
                samps = mkSamps parsed''
                samps' = shuffle_ samps g
                sampsTrain = P.drop 30 samps'
                sampsTest = P.take 30 samps'
                opt = RMSOptim { rmsoAlpha = constant 0.001, rmsoGamma = constant 0.9, rmsoGradAvg = AccMat (use (A.fromList (Z:.1:.1) [1.0])) Outp Inp}
                opt' = RMSOptim { rmsoAlpha = constant 0.001, rmsoGamma = constant 0.9, rmsoGradAvg = AccMat (use (A.fromList (Z:.1:.1) [1.0])) Outp One}
                optAnn = mkOptimANN ann opt opt'
                trainerFn = runN (trainer bli)
                (AccBlock obl _) = annOptim2block optAnn
            trained <- trainer' trainerFn (run obl) sampsTrain
            let firstAttemptFn = runN (\x -> let (_, AccMat o _ _) = inferANN ann (AccMat x Inp One) in o)
                firstAttempt = tester firstAttemptFn sampsTest
                trained' = P.snd (splitOptimANN (block2annOptim bli (AccBlock (use trained) (ANNBlock, Inp, Outp)) :: OptimANN RMSOptim RMSOptim Double Inp Outp))
                trained'' = AccBlock (use (run (let (AccBlock f _) = P.snd (ann2block trained') in f))) (ANNBlock, Inp, Outp)
                secondAttemptFn = runN (\x -> let (_, (AccMat outp _ _)) = inferANN (block2ann bli trained'') (AccMat x Inp One) in outp) 
                secondAttempt = tester secondAttemptFn sampsTest
            if secondAttempt P.>= firstAttempt 
            then do
                putStrLn ("Failed. Got: first attempt: " P.++ (show firstAttempt) P.++ " second attempt: " P.++ (show secondAttempt) )
                exitFailure
            else
                putStrLn ("Success. Got: first attempt: " P.++ (show firstAttempt) P.++ " second attempt: " P.++ (show secondAttempt))
