module Main where

import Control.DeepSeq
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Data.Array.Accelerate.Interpreter as PTX
import GHC.Conc
import ML.AI.ActFuncs
import ML.AI.ErrFn
import ML.AI.Layer
import ML.AI.Optim
import ML.AI.Optim.GD
import Prelude as P
import System.Exit
import System.IO
import System.Random

mkPerfectLine :: Double -> Double -> Int -> [(Double, Double)]
mkPerfectLine start end numSamples = do
    let nums = P.take numSamples [0..]
        numsD = P.map (\x -> P.fromIntegral x :: Double) nums
        numsD' = P.map (\x -> (end - start) * (x / (P.fromIntegral numSamples :: Double)) + start) numsD
    P.zip numsD' numsD'

sigmoidD :: Double -> Double
sigmoidD x = 1.0 / (1.0 + (exp (-x)))

testLine :: (Matrix Double -> Matrix Double ) -> [(Double, Double)] -> Double
testLine _ [] = 0.0
testLine fn ((x, y) : rest) = do
    let [predicted] = A.toList (fn (A.fromList (Z:.1:.1) [x])) 
        predicted' = sigmoidD predicted
    ((y - predicted') * (y - predicted')) + (testLine fn rest)

accTrainer :: ErrFn -> Acc (Vector Int, Vector Double) -> Acc (Vector Int, Vector Double) -> Acc (Vector Int, Vector Double) -> Acc (Matrix Double, Matrix Double) -> Acc ((Vector Int, Vector Double), (Vector Int, Vector Double), (Vector Int, Vector Double))
accTrainer efn block optimBlockW optimBlockB sample = do
    let (inp, outp) = A.unlift sample
        outpA = AccMat outp Outp One
        (layer, _) = block2l' block Inp Outp
        x = inferLayer layer (AccMat inp Inp One)
        e = derrFn efn outpA (actFunc Sigmoid x)
        woptim = block2optim optimBlockW :: GradDescOptim Double Outp Inp
        boptim = block2optim optimBlockB :: GradDescOptim Double Outp One
        lw = lweights layer
        lb = lbiases layer
        dw = ((matTransp lw) `matMul` (dactFunc Sigmoid x)) `matMul` (matTransp e)
        (woptim', lw') = optim woptim lw (matTransp dw)
        (boptim', lb') = optim boptim lb e
        block' = l2block' (Layer { lweights = lw', lbiases = lb' })
    A.lift (block', optim2block woptim', optim2block boptim')


trainer :: ((Vector Int, Vector Double) -> (Vector Int, Vector Double) -> (Vector Int, Vector Double) -> (Matrix Double, Matrix Double) -> ((Vector Int, Vector Double), (Vector Int, Vector Double), (Vector Int, Vector Double))) -> (Vector Int, Vector Double) -> (Vector Int, Vector Double) -> (Vector Int , Vector Double) -> [(Matrix Double, Matrix Double)] -> (Vector Int, Vector Double)
trainer _ p _ _ [] = p
trainer tfn params optimW optimB (h : t)= do
    let (params', optimW', optimB') = tfn params optimW optimB h
    (trainer tfn params' optimW' optimB' t)
    
filter' :: (a -> Bool) -> [a] -> ([a], [a])
filter' _ [] = ([], [])
filter' fn ( h : t) | fn h = do
    let (a, b) = filter' fn t
    (h : a, b)
filter' fn ( h : t) = do
    let (a, b) = filter' fn t
    (a, h : b)

sorter :: (P.Ord a, NFData a, NFData b ) => [(a, b)] -> [(a, b)]
sorter [] = []
sorter [a] = [a]
sorter ((h0, h1) : t) = do
    let (a, b) = force (filter' (\(c, _) -> c P.> h0) t)
        a' = sorter a
        b' = sorter (b P.++ [(h0, h1)])
    par b' (a' P.++ b')

shuffle :: (P.Ord a, NFData a, NFData b) => [a] -> [b] -> [b]
shuffle nums rets = do
    let (_, ret) = P.unzip (sorter (P.zip nums rets))
    ret

main :: IO ()
main = do
    let seed = mkStdGen 100
        p = Layer { lweights = (AccMat (use (A.fromList (Z:.1:.1) [10.0])) Outp Inp), lbiases = (AccMat (use (A.fromList (Z:.1:.1) [9.0])) Outp One)}
        fn = PTX.runN (\x -> let (AccMat y _ _ ) = inferLayer p (AccMat x Inp One) in y)
        rands = randoms seed :: [Int]
        perfectLine = force (mkPerfectLine (-10.0) 10.0 2048)
        perfectLine' = shuffle rands (P.map (\(x, y) -> (A.fromList (Z:.1:.1) [x], A.fromList (Z:.1:.1) [y])) perfectLine)
        testResults = testLine fn perfectLine 
        optimizerW = PTX.run (optim2block (GradDescOptim { gdoLR = (constant 0.001 :: Exp Double)}))
        optimizerB = PTX.run (optim2block (GradDescOptim { gdoLR = (constant 0.001 :: Exp Double)}))
        trainFn = PTX.runN (accTrainer MSEFn)
        newModel = trainer trainFn (PTX.run (l2block' p)) optimizerW optimizerB (force perfectLine')
        (newModelLayer, _) = block2l' (use newModel) Inp Outp 
        fn' = PTX.runN (\x -> let (AccMat m _ _ ) = inferLayer newModelLayer (AccMat x Inp One) in m)
        testResults' = testLine fn' perfectLine
    if testResults' P.< testResults
    then
        putStrLn (" GdOptim test succeeded. Got error0: " P.++ (show testResults) P.++ ", error1: " P.++ (show testResults') P.++ (show newModel))
    else do
        putStrLn (" GdOptim test failed. Got error0: " P.++ (show testResults) P.++ ", error1: " P.++ (show testResults') P.++ (show newModel))
        exitFailure
