module Main where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Data.Array.Accelerate.LLVM.PTX as PTX
import ML.AI.Layer
import ML.AI.Optim.GD
import ML.AI.Params
import Prelude as P
import System.Random

mkPerfectLine :: Double -> Double -> Int -> [(Double, Double)]
mkPerfectLine start end numSamples = do
    let nums = P.take numSamples [0..]
        numsD = P.map (\x -> P.fromIntegral x :: Double) nums
        numsD' = P.map (\x -> (x / (end - start)) + start) numsD
    P.zip numsD' numsD'

testLine :: (Matrix Double -> Matrix Double ) -> [(Double, Double)] -> Double
testLine _ [] = 0.0
testLine fn ((x, y) : rest) = let [predicted] = A.toList (fn (A.fromList (Z:.1:.1) [x])) in ((y - predicted) * (y - predicted)) + (testLine fn rest)

main :: IO ()
main = do
    let seed = mkStdGen 100
        p = Params { pweights = (AccMat (use (A.fromList (Z:.1:.1) [10.0])) Outp Inp), pbiases = (AccMat (use (A.fromList (Z:.1:.1) [9.0])) Outp One)}
        fn = PTX.runN (\x -> let (AccMat y _ _ ) = inferLayer p (AccMat x Inp One) in y)
        perfectLine = mkPerfectLine (-10.0) 10.0 50 
    putStrLn (show (testLine fn perfectLine))
