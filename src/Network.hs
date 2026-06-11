{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Network where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, Nat, natVal)
import Numeric.LinearAlgebra.Static
import System.IO.Unsafe (unsafePerformIO)
import System.Random (newStdGen, randoms)
import Unsafe.Coerce (unsafeCoerce)

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
    NilLayer :: (KnownNat i, KnownNat o) => Layer i o -> Layers i o
    ConsLayer :: (KnownNat i, KnownNat h) => Layer i h -> Layers h o -> Layers i o

makeLayers :: (KnownNat h, KnownNat o) => [Layer h h] -> Layer h o -> Layers h o
makeLayers ls out = foldr ConsLayer (NilLayer out) ls

mapLayer :: (forall i o. Layer i o -> Layer i o) -> Layers i o -> Layers i o
mapLayer f (NilLayer l) = NilLayer (f l)
mapLayer f (ConsLayer l rest) = ConsLayer (f l) (mapLayer f rest)

addLayers :: (KnownNat i, KnownNat o) => Layers i o -> Layers i o -> Layers i o
addLayers (NilLayer l) (NilLayer l') = NilLayer (l + l')
addLayers (ConsLayer l ls) (ConsLayer l' ls') =
    ConsLayer (l + unsafeCoerce l') (addLayers ls (unsafeCoerce ls'))
addLayers _ _ = error "addLayers: shape mismatch (impossible if construction is correct)"

class AppendLayer a b result | a b -> result where
    appendLayer :: a -> b -> result

instance (KnownNat i, KnownNat h) => AppendLayer (Layer i h) (Layers h o) (Layers i o) where
    appendLayer = ConsLayer

instance (KnownNat i, KnownNat h, KnownNat o) => AppendLayer (Layer i h) (Layer h o) (Layers i o) where
    appendLayer l l' = ConsLayer l (NilLayer l')

infixr 5 >->
(>->) :: (AppendLayer a b result) => a -> b -> result
(>->) = appendLayer

{-# NOINLINE randLayer #-}
randLayer :: forall i o. (KnownNat i, KnownNat o) => Activation -> Layer i o
randLayer a =
    let vals = map (\x -> 2 * x - 1) . randoms . unsafePerformIO $ newStdGen :: [Double]
        ni = fromIntegral (natVal (Proxy :: Proxy i))
        no = fromIntegral (natVal (Proxy :: Proxy o))
        w = matrix (take (ni * no) vals)
        b = vector (take no (drop (ni * no) vals))
     in Layer w b a

relu :: Activation
relu = Activation (max 0) (\x -> if x > 0 then 1 else 0)

linear :: Activation
linear = Activation id (const 1.0)

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

scaleLayers :: (KnownNat o) => Double -> Layers i o -> Layers i o
scaleLayers eta (NilLayer (Layer w b a)) = NilLayer (Layer (dmmap (* eta) w) (dvmap (* eta) b) a)
scaleLayers eta (ConsLayer (Layer w b a) rest) = ConsLayer (Layer (dmmap (* eta) w) (dvmap (* eta) b) a) (scaleLayers eta rest)

backprop :: (KnownNat i, KnownNat o) => R i -> R o -> Layers i o -> GradCost o -> Layers i o
backprop x y layers c' =
    let (yHat, backward) = forward layers x
        deltaL = c' yHat y
        (_, grads) = backward deltaL
     in grads
