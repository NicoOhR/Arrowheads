{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module Main where

import Control.Monad (foldM)
import Control.Monad.Writer
import qualified Data.Aeson as A
import Data.Bifunctor
import qualified Data.ByteString.Lazy as BL
import qualified Data.DList as DL
import Graphics.Vega.VegaLite hiding (Theta)
import Numeric.LinearAlgebra
import System.Random
import Prelude hiding ((<>))

-- n, m, width, length
data Dimensions = Dimensions Int Int Int Int

-- Cost function type signature
type GradCost = Vector Double -> Vector Double -> Vector Double

-- Activation function type signature
type Activation = Double -> Double

-- Model parameters W_i, b_i
data Theta where
    Theta :: [(Matrix Double, Vector Double)] -> Theta
    deriving (Show)

-- Parameters, Cost derivative, Activation, Activation derivative
data Network = Network Theta GradCost Activation Activation

-- need to implement a vector space typeclass for this
sumThetas :: Theta -> Theta -> Theta
sumThetas (Theta ts) (Theta gs) = Theta (zipWith (\(x, y) (u, v) -> (x + u, y + v)) ts gs)

subtractThetas :: Theta -> Theta -> Theta
subtractThetas (Theta ts) (Theta gs) = Theta (zipWith (\(x, y) (u, v) -> (x - u, y - v)) ts gs)

scaleThetas :: Theta -> Double -> Theta
scaleThetas (Theta ts) eta = Theta (fmap (bimap (scale eta) (scale eta)) ts)

relu :: Activation
relu x = max x 0

relu' :: Activation
relu' x = if x > 0 then 1 else 0

gradEuclidean :: GradCost
gradEuclidean x y = 2 * (x - y)

forwardW ::
    Network ->
    Vector Double ->
    Writer (DL.DList (Vector Double)) (Vector Double)
forwardW (Network (Theta ts) _ f _) x1 = do
    let (hiddenLayer, lastLayer) = (init ts, last ts)
    h <- foldM go x1 hiddenLayer
    let (wL, bL) = lastLayer
        y_hat = wL #> h + bL
    tell (DL.singleton y_hat)
    pure y_hat
  where
    go :: Vector Double -> (Matrix Double, Vector Double) -> Writer (DL.DList (Vector Double)) (Vector Double)
    go x (w, b) = do
        let z = cmap f (w #> x + b)
        tell (DL.singleton z)
        pure z

backprop :: Vector Double -> Vector Double -> Network -> Theta
backprop x y (Network (Theta ts) c' f f') =
    let
        (y_hat, zsD) = runWriter (forwardW (Network (Theta ts) c' f f') x)
        as = DL.toList zsD -- [a1, a2 .., aL]
        activations = init (x : as) -- [a0, a1, a2 ... a(L-1)]
        derivatives = map (cmap f') (init as) -- [f'(a1), f'(a2)..., f'(aL-1)]
        deltaL = c' y_hat y
        deltas = scanr go deltaL (zip (tail ts) derivatives) -- [((w2, b2), f'(a1))...]
        go :: ((Matrix Double, Vector Double), Vector Double) -> Vector Double -> Vector Double
        go ((w, _), z) d = (tr w #> d) * z
        gradw = zipWith (\a d -> asColumn d <> asRow a) activations deltas
     in
        Theta (zip gradw deltas)

initNetwork :: (RandomGen g) => Dimensions -> g -> Theta
initNetwork (Dimensions n m width len) gen = Theta ([input] ++ hidden ++ [output])
  where
    getList t = take t $ uniformRs (-1.0 :: Double, 1.0 :: Double) gen
    input = ((width >< n) $ getList (n * width), fromList $ getList width) -- n x width
    hidden = replicate len ((width >< width) $ getList (width * width), fromList $ getList width) -- width x width length times
    output = ((m >< width) $ getList (m * width), fromList $ getList m) -- width x n

train :: Network -> [(Vector Double, Vector Double)] -> Network
train nn [] = nn
train (Network (Theta theta) c' f f') ((x, y) : ys) = train (Network iterated c' f f') ys
  where
    iterated = subtractThetas (Theta theta) (scaleThetas (backprop x y (Network (Theta theta) c' f f')) 0.00001)

xorData :: [(Vector Double, Vector Double)]
xorData =
    [ (fromList [0, 0], fromList [0])
    , (fromList [0, 1], fromList [1])
    , (fromList [1, 0], fromList [1])
    , (fromList [1, 1], fromList [0])
    ]

xorDataRepeated :: Int -> [(Vector Double, Vector Double)]
xorDataRepeated k = concat (replicate k xorData)

sinData :: (RandomGen g) => Int -> g -> [(Vector Double, Vector Double)]
sinData k gen = zip x y
  where
    x = fmap scalar $ take k $ uniformRs (0 :: Double, 2 * pi :: Double) gen
    y = fmap sin x

sinTest :: [(Double, Double)]
sinTest = zip x y
  where
    x = [0, 0.01 .. 2 * pi]
    y = map sin x

main :: IO ()
main = do
    let
        gen = mkStdGen 10
        dims = Dimensions 1 1 5 7
        theta = initNetwork dims gen
        net = Network theta gradEuclidean relu relu'
        trained = train net (sinData 100000 gen)
        test = fmap (fst . runWriter . forwardW trained) (map (scalar . fst) sinTest)
        test_double = zip [0, 0.01 .. 2 * pi] (fmap (! 0) test)
        mse_list = fmap (\(x, y) -> (x - y) ** 2) (zip (map snd test_double) (map (sin . fst) test_double))
        mse = sum mse_list / fromIntegral (length mse_list)
        dat =
            dataFromColumns []
                . dataColumn "x" (Numbers (map fst test_double))
                . dataColumn "y_hat" (Numbers (map snd test_double))
                . dataColumn "y" (Numbers (map (sin . fst) test_double))
        bkg = background "rgba(0,0,0,0.05)"
        enc = encoding . position X [PName "x", PmType Quantitative] . position Y [PName "y", PmType Quantitative] . color [MName "Origin"]
        encBase field =
            encoding
                . position X [PName "x", PmType Quantitative]
                . position Y [PName field, PmType Quantitative]
    print mse
    BL.writeFile "output.json" $ A.encode (fromVL $ toVegaLite [bkg, dat [], layer [asSpec [mark Line [MStroke "steelblue"], encBase "y" []], asSpec [mark Line [MStroke "firebrick"], encBase "y_hat" []]], enc []])
