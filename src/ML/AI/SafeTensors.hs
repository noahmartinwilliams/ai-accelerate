{-# LANGUAGE FlexibleContexts #-}
module ML.AI.SafeTensors where

import Data.Array.Accelerate as A
import Data.Binary.IEEE754
import Data.Binary.Put 
import Data.ByteString.Lazy as BS

import Prelude as P

class ShowNType dt where
    showNType :: dt -> String
    storeData :: dt -> ByteString

instance ShowNType Double where
    showNType _ = "F64"
    storeData dt = runPut (putFloat64le dt) 


mat2safeTensor :: (ShowNType f) => String -> ([Int], [f]) -> (Int -> (ByteString, ByteString), ([Int], [f]))
mat2safeTensor name ([], _) = error ("Internal error: tensor \"" P.++ name P.++ "\" passed to mat2safeTensor with empty list.")
mat2safeTensor name (_, []) = error ("Internal error: tensor \"" P.++ name P.++ "\" passed to mat2safeTensor with empty list.")
mat2safeTensor name ((m : 1 : rest) , ds) | (P.length ds) P.>= m = do
    let matDs = P.take m ds
        meta = "\"" P.++ name P.++ "\":{\"dtype\":\"" P.++ (showNType (matDs P.!! 0)) P.++ "\",\"shape\":[" P.++ (show m) P.++ "],\"data_offsets\":["
        tensors = P.map storeData matDs
        tensors' = P.foldr BS.append BS.empty tensors
    ((\i -> (fromString (meta P.++ (show i) P.++ "," P.++ (show (i + (P.fromIntegral (BS.length tensors')))) P.++ "]}"), tensors')), (rest, P.drop m ds)) 
mat2safeTensor name ((m : n : rest), ds) | (P.length ds) P.>= (m * n) = do
    let matDs = P.take (m * n) ds
        meta = "\"" P.++ name P.++ "\":{\"dtype\":\"" P.++ (showNType (matDs P.!! 0)) P.++ "\",\"shape\":[" P.++ (show m) P.++ "," P.++ (show n) P.++ "],\"data_offsets\":["
        tensors = P.map storeData matDs
        tensors' = P.foldr BS.append BS.empty tensors
    ((\i -> (fromString (meta P.++ (show i) P.++ "," P.++ (show (i + (P.fromIntegral (BS.length tensors')))) P.++ "]}"), tensors')), (rest, P.drop (m * n) ds))
mat2safeTensor name _ = error ("internal error: tensor \"" P.++ name P.++ "\" has malformed lists in mat2safeTensor")

block2safeTensors' :: (ShowNType f) => Int -> String -> ([Int], [f]) -> ([(Int -> (ByteString, ByteString))], ([Int], [f]))
block2safeTensors' nlayers nm bl = intern nlayers 0 nm bl where
    intern :: (ShowNType f) => Int -> Int -> String -> ([Int], [f]) -> ([(Int -> (ByteString, ByteString))], ([Int], [f]))
    intern 0 _ _ bl' = ([], bl')
    intern numLayers n name block = do
        let nameMat = (name P.++ "$weights$" P.++ (show n))
            nameBias = (name P.++ "$biases$" P.++ (show n))
            (wmat, block') = mat2safeTensor nameMat block
            (bmat, block'') = mat2safeTensor nameBias block'
            (rest, block2) = intern (numLayers - 1) (n + 1) name block''
        (wmat : bmat : rest, block2)

block2safeTensors :: (ShowNType f) => Int -> String -> ([Int], [f]) -> ByteString
block2safeTensors nlayers nm bl = do
    let (fnList, _) = block2safeTensors' nlayers nm bl
        (meta, tensors) = intern fnList 0
        metaL = BS.length meta
    (runPut (putWord64le (P.fromIntegral metaL))) `BS.append` (fromString "{") `BS.append` meta `BS.append` (fromString "}") `BS.append` tensors where
        intern :: [(Int -> (ByteString, ByteString))] -> Int -> (ByteString, ByteString)
        intern [] _ = (BS.empty, BS.empty)
        intern [fn] n = do
            let (meta, tensor) = fn n 
            (meta , tensor )
        intern (fn : r) n = do
            let (meta, tensor) = fn n 
                n' = P.fromIntegral (BS.length tensor)
                (restM, restT) = intern r n'
            (meta `BS.append` (fromString "," ) `BS.append` restM, tensor `BS.append` restT)
