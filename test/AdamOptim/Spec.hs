{-# LANGUAGE FlexibleContexts, TypeOperators #-}
module Main where


import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.PTX
import Data.Array.Accelerate.Matrix
import Data.Array.Accelerate.Debug.Trace

import Data.ByteString as BS
import Data.ByteString.Lazy as BSL

import Data.Text as T

import ML.AI.ActFuncs
import ML.AI.ANN
import ML.AI.Block
import ML.AI.ErrFn
import ML.AI.Grad
import ML.AI.Layer
import ML.AI.OptimANN
import ML.AI.Optim.Adam
import ML.AI.Optim.Mom
import ML.AI.Optim.RMS
import ML.AI.SafeTensors
import ML.AI.Types

import Prelude as P
import System.Exit
import System.IO
import System.Random

import Text.Printf

data BatchSize = BatchSize
data Grad = Grad

mkBS :: Int -> BS.ByteString -> BS.ByteString -> [(BS.ByteString , BS.ByteString)]
mkBS numImgsPerSamp imgBS labelBS = do
    let imgFormat = BS.drop 16 imgBS -- Skip metadata.
        labelFormat = BS.drop 8 labelBS
    intern numImgsPerSamp imgFormat labelFormat where
        intern :: Int -> BS.ByteString -> BS.ByteString -> [(BS.ByteString, BS.ByteString)]
        intern _ bs _ | BS.null bs = []
        intern n imgs labels = do
            let imgs' = BS.take (n * 28 * 28) imgs
                labels' = BS.take n labels
                imgsNext = BS.drop (n * 28 * 28) imgs
                labelsNext = BS.drop n labels
            (imgs', labels') : (intern n imgsNext labelsNext)

mkSamps :: Int -> [(BS.ByteString, BS.ByteString)] -> [(Matrix Double, Matrix Double)] 
mkSamps numImgsPerSamp ls = do
    let (imgs, labels) = P.unzip ls
        labels' = getLabels numImgsPerSamp labels 
        imgs' = getImgs numImgsPerSamp imgs
    P.zip imgs' labels' where

        getImgs :: Int -> [BS.ByteString] -> [Matrix Double]
        getImgs _ [] = []
        getImgs n ( h : t ) = (A.fromList (Z:.(28*28):.n) (P.map (\x -> ((P.fromIntegral x :: Double) / 255.0)) (BS.unpack h))) : (getImgs n t)

        getLabels :: Int -> [BS.ByteString] -> [Matrix Double]
        getLabels _ [] = []
        getLabels n (h : t) = do
            let labels = getLabels' h
                labels' = A.fromList (Z:.10:.n) (P.foldr (P.++) [] labels) :: Matrix Double
            labels' : (getLabels n t)

        getLabels' :: BS.ByteString -> [[Double]]
        getLabels' bs | BS.null bs = []
        getLabels' bs = do
            let bs' = (BS.unpack (BS.take 1 bs)) P.!! 0
            let numV = case bs' of
                    0 -> [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    1 -> [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    2 -> [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    3 -> [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    4 -> [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    5 -> [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                    6 -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    7 -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                    8 -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    9 -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    _ -> error "Invalid number!"
            numV : (getLabels' (BS.drop 1 bs))

tester :: ((Matrix Double) -> (Matrix Double)) -> [(Matrix Double, Matrix Double)] -> Double
tester _ [] = 0.0
tester fn ( h : t) = do
    let outp = fn (P.fst h)
        outp' = A.toList outp
        ret = (P.sum (P.zipWith (\x -> \y -> (x - y) * (x - y)) (A.toList (P.snd h)) outp' )) 
    ret + (tester fn t)

trainFn :: Int -> ANNBlInfo -> Acc (Scalar Double, (Vector Int, Vector Double)) -> Acc (Matrix Double, Matrix Double) -> Acc (Scalar Double, (Vector Int, Vector Double))
trainFn batchSize bli optimV sample = do
    let (inp, outp) = A.unlift sample :: (Acc (Matrix Double), Acc (Matrix Double))
        (_, optimV') = A.unlift optimV :: (Acc (Scalar Double), Acc (Vector Int, Vector Double))
        optimizer = block2annOptim bli (AccBlock optimV' (ANNBlock, Inp, Outp))
        (_, ann) = splitOptimANN optimizer
        nullg = A.unlift $ getNullGrads optimizer :: (Acc (Vector Int), Acc (Vector Double))
        nullErr = use (A.fromList (Z) [0.0] :: Scalar Double)
        final = (awhile condition (trainFn' batchSize ann) (A.lift (inp, outp, nullg, nullErr)) :: Acc (Matrix Double, Matrix Double, (Vector Int, Vector Double), Scalar Double))
        (_, _, grads, bpe) = A.unlift final :: (Acc (Matrix Double), Acc (Matrix Double), Acc (Vector Int, Vector Double), Acc (Scalar Double))
        optim' = bpANN optimizer (AccBlock grads Inp) 
        (AccBlock optim'' _) = annOptim2block optim'
    A.lift (bpe, optim'') where

        condition :: Acc (Matrix Double, Matrix Double, (Vector Int, Vector Double), Scalar Double) -> Acc (Scalar Bool)
        condition t = do
            let (inp'', _, _, _) = A.unlift t :: (Acc (Matrix Double), Acc (Matrix Double), Acc (Vector Int, Vector Double), Acc (Scalar Double))
            (A.unit ((A.null inp'') A.== (constant False)))

        getNullGrads :: OptimANN AdamOptim AdamOptim Double Inp Outp -> Acc (Vector Int, Vector Double) 
        getNullGrads (OptimANN layers _ _) = do
            let (AccBlock ret _) = P.foldr (\x -> \y -> appendAccBlocks x y Grad) (accEmptyBlock Grad) (P.map getNullGradsLayer layers)
            ret

        getNullGradsLayer :: OptimANNLayer AdamOptim AdamOptim Double Inp Outp -> AccBlock Double Grad
        getNullGradsLayer (OptimANNLayer { oalLayer = ANNLayer { annL = Layer { lweights = w, lbiases = b}}}) = do
            let w' = accMatGetBlock (matMap w (\_ -> constant 0.0)) Grad
                b' = accMatGetBlock (matMap b (\_ -> constant 0.0)) Grad
            appendAccBlocks w' b' Grad

trainFn' :: Int -> ANN Double Inp Outp -> Acc (Matrix Double , Matrix Double, (Vector Int, Vector Double), Scalar Double) -> Acc (Matrix Double, Matrix Double, (Vector Int, Vector Double), Scalar Double)
trainFn' bsize ann input = do
    let (inp, outp, grads, bpe) = A.unlift input :: (Acc (Matrix Double), Acc (Matrix Double), Acc (Vector Int, Vector Double), Acc (Scalar Double))
        (prevIns, outp') = inferANN ann (AccMat (A.take (constant 1) inp) Inp One)
        (AccMat bp Inp One) = (derrFn LogEFn ) (AccMat (A.take (constant 1) outp) Inp One) outp'
        (AccMat bpe' Inp One) = (errFn LogEFn) (AccMat (A.take (constant 1) outp) Inp One) outp'
        (grads', _) = gradANN bsize ann prevIns (AccMat bp Outp One)
        gradBlock = grad2block grads' Grad
        (AccBlock gradBlock' _) = gradAddDs (AccBlock grads Grad) gradBlock
        inp' = A.drop (constant 1) ( inp)
        outp'' = A.drop (constant 1) ( outp)
    (A.lift (inp', outp'', gradBlock', (A.zipWith (+) bpe (A.sum (A.flatten bpe') ))))

type TrainFn = (Scalar Double, (Vector Int, Vector Double)) -> (Matrix Double, Matrix Double) -> (Scalar Double, (Vector Int, Vector Double))

getTrained :: TrainFn -> [(Matrix Double, Matrix Double)] -> (Scalar Double, (Vector Int, Vector Double)) -> IO (Vector Int, Vector Double)
getTrained _ [] (err, outp) = do
    putStrLn (printf "error: %.5f" ((A.toList err) P.!! 0) :: String) 
    return outp
getTrained tfn (h : t) inp = do
    let (err, outp) = tfn inp h
    putStrLn (printf "error: %.5f" ((A.toList err) P.!! 0) :: String) 
    System.IO.appendFile "/tmp/errors.txt" ((printf "%.5f" ((A.toList err) P.!! 0) :: String) P.++ "\n")
    getTrained tfn t (err, outp)

main :: IO ()
main = do
    trainImgs <- BS.readFile "data/mnist/train-images.idx3-ubyte"
    trainLabels <- BS.readFile "data/mnist/train-labels.idx1-ubyte"
    testImgs <- BS.readFile "data/mnist/t10k-images.idx3-ubyte"
    testLabels <- BS.readFile "data/mnist/t10k-labels.idx1-ubyte"
    let batchSize = 60
        g = mkStdGen 100
        bsTrain = mkBS batchSize trainImgs trainLabels
        sampsTrain = (mkSamps batchSize bsTrain) 
        bsTest = mkBS 1 testImgs testLabels
        sampsTest = (mkSamps 1 bsTest)
        (bli, ann) = mkANN g (28*28) ReLU [(32, ReLU), (10, SoftMax)] Inp Outp
        alpha = constant 0.001
        b1 = constant 0.9
        b2 = constant 0.999
        basicMat = use (A.fromList (Z:.1:.1) [1.0])
        optW = AdamOptim { adamoBeta1t = b1, adamoBeta2t = b2, adamoMomentum = MomentOptim { momoAlpha = alpha, momoBeta = b1, momoVelocity = AccMat basicMat Outp Inp}, adamoRMS = RMSOptim { rmsoAlpha = alpha, rmsoGamma = b2, rmsoGradAvg = AccMat basicMat Outp Inp}}
        optB = AdamOptim { adamoBeta1t = b1, adamoBeta2t = b2, adamoMomentum = MomentOptim { momoAlpha = alpha, momoBeta = b1, momoVelocity = AccMat basicMat Outp One}, adamoRMS = RMSOptim { rmsoAlpha = alpha, rmsoGamma = b2, rmsoGradAvg = AccMat basicMat Outp One}}
        optimANN = mkOptimANN ann optW optB
        testFn testable = runN (\x -> let (_, AccMat y _ _) = inferANN testable (AccMat x Inp One) in y)
        testedInitial = tester (testFn ann) sampsTest
        (AccBlock optimBlock _) = annOptim2block optimANN
        trainer = runN (trainFn batchSize bli)
        optimBlockRan = run optimBlock
    finalTrained <- (getTrained trainer (P.take 600000 (P.foldr (P.++) [] (P.repeat sampsTrain))) (A.fromList (Z) [0.0], optimBlockRan))
    let annFinal = block2annOptim bli (AccBlock (use finalTrained) (ANNBlock, Inp, Outp)) :: OptimANN AdamOptim AdamOptim Double Inp Outp
        (_, annFinal') = splitOptimANN annFinal
        testedFinal = tester (testFn annFinal') sampsTest
        (_, (AccBlock annFinal'' _)) = ann2block annFinal'
        annFinal2 = run annFinal''
    if testedInitial P.> testedFinal
    then do
        putStrLn ("\nSuccess. Initial error: " P.++ (P.show testedInitial) P.++ ". Final error: " P.++ (P.show testedFinal))
        BSL.writeFile "adam.safetensor" (block2safeTensors 3 "adamANN" (A.toList (P.fst annFinal2), A.toList (P.snd annFinal2)))
    else do
        putStrLn ("\nFailure. Initial error: " P.++ (P.show testedInitial) P.++ ". Final error: " P.++ (P.show testedFinal))
        exitFailure
