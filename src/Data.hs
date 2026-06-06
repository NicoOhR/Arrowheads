{-# LANGUAGE DataKinds #-}
module Data where

import Numeric.LinearAlgebra.Static
import System.Random

xorData :: [(R 2, R 1)]
xorData =
    [ (vector [0, 0], vector [0])
    , (vector [0, 1], vector [1])
    , (vector [1, 0], vector [1])
    , (vector [1, 1], vector [0])
    ]

xorDataRepeated :: Int -> [(R 2, R 1)]
xorDataRepeated k = concat (replicate k xorData)

sinData :: RandomGen g => Int -> g -> [(R 1, R 1)]
sinData k gen = zip xs ys
  where
    vals = take k $ uniformRs (0 :: Double, 2 * pi) gen
    xs   = fmap (vector . pure) vals
    ys   = fmap ((vector . pure) . sin) vals

sinDataScaled :: RandomGen g => Int -> g -> [(R 1, R 1)]
sinDataScaled k gen = zip xs ys
  where
    vals = take k $ uniformRs (0 :: Double, 2 * pi) gen
    xs   = fmap ((vector . pure) . (/ (2 * pi))) vals
    ys   = fmap ((vector . pure) . sin) vals

sinTest :: [(R 1, R 1)]
sinTest = zip xs ys
  where
    vals = [0, 0.01 .. 2 * pi]
    xs   = fmap ((vector . pure) . (/ (2 * pi))) vals
    ys   = fmap ((vector . pure) . sin) vals
