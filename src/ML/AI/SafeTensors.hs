{-# LANGUAGE FlexibleContexts #-}
module ML.AI.SafeTensors where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
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

mat2safeTensor :: (ShowNType f, A.Floating f) => String -> ([Int], [f]) -> (Int -> (ByteString, ByteString, ([Int], [f])))
mat2safeTensor name ([], _) = error ("Internal error: tensor \"" P.++ name P.++ "\" passed to mat2safeTensor with empty list.")
mat2safeTensor name (_, []) = error ("Internal error: tensor \"" P.++ name P.++ "\" passed to mat2safeTensor with empty list.")
mat2safeTensor name ((m : 1 : rest) , ds) | (P.length ds) P.>= m = do
    let matDs = P.take m ds
        meta = "\"" P.++ name P.++ "\":{\"dtype\":\"" P.++ (showNType (matDs P.!! 0)) P.++ "\",\"shape\":[" P.++ (show m) P.++ "],\"data_offsets\":["
        tensors = P.map storeData matDs
        tensors' = P.foldr BS.append BS.empty tensors
    (\i -> (fromString (meta P.++ (show i) P.++ "," P.++ (show (i + (P.fromIntegral (BS.length tensors')))) P.++ "]}"), tensors', (rest, P.drop m ds) ))
mat2safeTensor name ((m : n : rest), ds) | (P.length ds) P.>= (m * n) = do
    let matDs = P.take (m * n) ds
        meta = "\"" P.++ name P.++ "\":{\"dtype\":\"" P.++ (showNType (matDs P.!! 0)) P.++ "\",\"shape\":[" P.++ (show m) P.++ "," P.++ (show n) P.++ "],\"data_offsets\":["
        tensors = P.map storeData matDs
        tensors' = P.foldr BS.append BS.empty tensors
    (\i -> (fromString (meta P.++ (show i) P.++ "," P.++ (show (i + (P.fromIntegral (BS.length tensors')))) P.++ "]}"), tensors', (rest, P.drop (m * n) ds) ))

mat2safeTensor name _ = error ("internal error: tensor \"" P.++ name P.++ "\" has malformed lists in mat2safeTensor")
