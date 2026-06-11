{-# LANGUAGE DataKinds #-}

module Main where

import qualified Data.Vector.Storable as V
import Numeric.LinearAlgebra.Static (R, extract)
import System.Random (mkStdGen)

import Data
import Network

trainStep :: Double -> GradCost 1 -> Layers 1 1 -> (R 1, R 1) -> Layers 1 1
trainStep eta c' net (x, y) =
    let grads = backprop x y net c'
     in addLayers net (scaleLayers (-eta) grads)

trainEpoch :: Double -> GradCost 1 -> [(R 1, R 1)] -> Layers 1 1 -> Layers 1 1
trainEpoch eta c' dataset net = foldl (trainStep eta c') net dataset

main :: IO ()
main = do
    let eta = 0.0001
        network =
            (randLayer relu :: Layer 1 10)
                >-> (randLayer relu :: Layer 10 10)
                >-> (randLayer relu :: Layer 10 5)
                >-> (randLayer linear :: Layer 5 1)
        datasets = replicate 100 (sinDataScaled 30000 (mkStdGen 42))
        trained = scanl (flip (trainEpoch eta gradEuclidean)) network datasets
        testXs = map fst sinTest
        results = map (\net -> map (V.head . extract . fst . forward net) testXs) trained
        history = map (zip ([0, 0.01 .. 2 * pi] :: [Double])) results
    writeFile "output.txt" $ show history
