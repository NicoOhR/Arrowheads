{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Network where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, Nat, natVal)
import Numeric.LinearAlgebra.Static
import System.Random (mkStdGen, randoms)

-- Cost function type signature
type GradCost (n :: Nat) = R n -> R n -> R n

-- Activation function and its derivative
data Activation = Activation
    { act :: Double -> Double
    , act' :: Double -> Double
    }

data Layer (i :: Nat) (o :: Nat) where
    Layer :: L o i -> R o -> Activation -> Layer i o

instance (KnownNat i, KnownNat o) => Num (Layer i o) where
    (Layer w b a) + (Layer w' b' _) = Layer (w + w') (b + b') a
    (Layer w b a) - (Layer w' b' _) = Layer (w - w') (b - b') a

data Layers (i :: Nat) (o :: Nat) where
    NilLayer :: Layer i o -> Layers i o
    ConsLayer :: (KnownNat h) => Layer i h -> Layers h o -> Layers i o

makeLayers :: (KnownNat h) => [Layer h h] -> Layer h o -> Layers h o
makeLayers ls out = foldr ConsLayer (NilLayer out) ls

mapLayer :: (forall i o. Layer i o -> Layer i o) -> Layers i o -> Layers i o
mapLayer f (NilLayer l) = NilLayer (f l)
mapLayer f (ConsLayer l rest) = ConsLayer (f l) (mapLayer f rest)

randLayer :: forall i o. (KnownNat i, KnownNat o) => Int -> Activation -> Layer i o
randLayer seed a =
    let vals = map (\x -> 2 * x - 1) . randoms . mkStdGen $ seed :: [Double]
        ni = fromIntegral (natVal (Proxy :: Proxy i))
        no = fromIntegral (natVal (Proxy :: Proxy o))
        w = matrix (take (ni * no) vals)
        b = vector (take no (drop (ni * no) vals))
     in Layer w b a

relu :: Activation
relu = Activation (max 0) (\x -> if x > 0 then 1 else 0)

gradEuclidean :: GradCost n
gradEuclidean x y = 2 * (x - y)

forward :: (KnownNat i, KnownNat o) => Layers i o -> R i -> (R o, R o -> (R i, Layers i o))
forward (NilLayer (Layer w b (Activation a a'))) input =
    let z = w #> input + b
     in ( dvmap a z
        , \d ->
            let delta = d * dvmap a' z
             in ( tr w #> delta
                , NilLayer (Layer (outer delta input) delta (Activation a a'))
                )
        )
forward (ConsLayer (Layer w b (Activation a a')) rest) input =
    let z = w #> input + b
        aOut = dvmap a z
        (output, restBack) = forward rest aOut
     in ( output
        , \d ->
            let (deltaA, restGrad) = restBack d
                deltaZ = deltaA * dvmap a' z
                deltaInput = tr w #> deltaZ
             in ( deltaInput
                , ConsLayer (Layer (outer deltaZ input) deltaZ (Activation a a')) restGrad
                )
        )

backprop :: (KnownNat i, KnownNat o) => R i -> R o -> Layers i o -> GradCost o -> Layers i o
backprop x y layers c' =
    let (yHat, backward) = forward layers x
        deltaL = c' yHat y
        (_, grads) = backward deltaL
     in grads
