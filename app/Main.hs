{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

{-# HLINT ignore "Redundant bracket" #-}
module Main where

import Control.Monad (foldM)
import Control.Monad.Writer
import qualified Data.DList as DL
import Debug.Trace (trace, traceShow, traceShowId)
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import Prelude hiding ((<>))

-- n, m, width, length
data Dimensions = Dimensions Int Int Int Int

-- Cost function type signature, TODO needs to accept vector argument
type Cost = Vector Double -> Vector Double -> Double

-- Activation function type signature
type Activation = Double -> Double

-- Model parameters W_i, b_i
data Theta = Theta [(Matrix Double, Vector Double)] deriving (Show)

data Network = Network Dimensions Theta Cost Activation

sumThetas :: Theta -> Theta -> Theta
sumThetas (Theta ts) (Theta gs) = Theta (zipWith (\(x, y) (u, v) -> (x + u, y + v)) ts gs)

relu :: Activation
relu x = max x 0

relu' :: Activation
relu' x = if x > 0 then 1 else 0

euclidean :: Cost
euclidean x y = ((x - y) <.> (x - y))

gradEuclidean :: Vector Double -> Vector Double -> Vector Double
gradEuclidean x y = 2 * (x - y)

forward :: Network -> Vector Double -> Vector Double
forward (Network _ (Theta ts) _ f) x1 = foldl (iter) x1 ts
  where
    iter x (w, b) = cmap f (w #> x + b)

forwardW ::
    Network ->
    Vector Double ->
    Writer (DL.DList (Vector Double)) (Vector Double)
forwardW (Network _ (Theta ts) _ f) x1 = foldM (iter) x1 ts
  where
    iter ::
        Vector Double ->
        (Matrix Double, Vector Double) ->
        Writer (DL.DList (Vector Double)) (Vector Double)
    iter x (w, b) = do
        let z = cmap f (w #> x + b)
        writer (z, DL.singleton z)

backprop :: Vector Double -> Vector Double -> Network -> Theta
backprop x y (Network dims (Theta ts) cost f) =
    let
        (y_hat, zs) = runWriter (forwardW (Network dims (Theta ts) cost f) x)
        active = map (cmap relu') (DL.toList zs)
        activePrev = x : init active
        deltaL = gradEuclidean y_hat y * last active
        deltas = scanr go deltaL (zip (tail ts) (init active))
        go :: ((Matrix Double, Vector Double), Vector Double) -> Vector Double -> Vector Double
        go ((w, _), z) d = (tr w #> d) * z
        gradw = zipWith (\a d -> asColumn d <> asRow a) activePrev deltas
     in
        Theta (zip gradw deltas)

train :: Network -> [(Vector Double, Vector Double)] -> Network
train network ts = go network ts
  where
    go nn [] = nn
    go (Network d (Theta theta) cost f) ((y_hat, y) : ys) = go (Network d iterated cost f) ys
      where
        iterated = sumThetas (Theta theta) (backprop y_hat y (Network d (Theta theta) cost f))

main :: IO ()
main = do
    let w1 =
            (2 >< 1)
                [ 1.0
                , -0.5
                ]
        b1 = fromList [0.1, (-0.2)]
        w2 =
            (2 >< 2)
                [ 1.0
                , 0.0
                , 0.0
                , 1.0
                ]
        b2 = fromList [0.0, 0.3]
        w3 =
            (1 >< 2)
                [ 0.5
                , 0.5
                ]
        b3 = fromList [0.0]
        theta = Theta [(w1, b1), (w2, b2), (w3, b3)]
        dims = Dimensions 2 2 2 2
        net = Network dims theta euclidean relu

        xInput = fromList [1.0]

        output = forwardW net xInput
        grad = backprop xInput (fromList [2.0]) net
        training_x = map (fromList . (\x -> [x])) [0, 0.05 .. 2 * pi]
        training_y = map (fromList . (\x -> [x]) . sin) [0, 0.05 .. 2 * pi]
        training = zip training_x training_y
        (Network _ (Theta s) _ _) = train net training
    -- test = forwardW trained (pi / 4)
    print s
