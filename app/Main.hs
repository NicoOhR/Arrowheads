{-# LANGUAGE DataKinds #-}

module Main where

import qualified Data.Vector.Storable as V
import Numeric.LinearAlgebra.Static (extract)
import System.Random (mkStdGen)

import Data
import Network

initNetwork :: Layers 1 1
initNetwork =
    let input = randLayer 0 relu :: Layer 1 10
        hidden = map (\s -> randLayer s relu :: Layer 10 10) [2, 4 .. 20]
        output = randLayer 22 relu :: Layer 10 1
     in ConsLayer input (makeLayers hidden output)

main :: IO ()
main = do
    let datasets = replicate 100 (sinDataScaled 30000 (mkStdGen 42))
    undefined
