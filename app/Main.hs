module Main where
import Numeric.LinearAlgebra.Data
import Numeric.LinearAlgebra
import qualified Data.DList as DL
import Control.Monad.Writer
import Control.Monad (foldM)
-- n, m, width, length
data Dimensions = Dimensions Int Int Int Int
-- Cost function type signature, TODO needs to accept vector argument
type Cost = Vector Double -> Vector Double -> Double
-- Activation function type signature
type Activation = Double -> Double
-- Model parameters W_i, b_i
data Theta = Theta [(Matrix Double, Vector Double)]

data Network = Network Dimensions Theta Cost Activation 

relu :: Activation
relu x = if x > 0 then x else 0 

relu' :: Activation 
relu' x = if x > 0 then 1 else 0

euclidean :: Cost 
euclidean x y = ((x - y) <.> (x -y))

grad_euclidean :: Vector Double -> Vector Double -> Vector Double
grad_euclidean x y = 2 * (x - y)

-- need to track this intermediate z with writer monad
forward :: Network -> Vector Double -> Vector Double
forward (Network _ (Theta ts) _ f) x1 = foldl (iter) x1 ts where 
    iter x (w, b) = cmap f (w #> x + b)

forward_intermediate :: Network 
                     -> Vector Double 
                     -> Writer (DL.DList (Vector Double)) (Vector Double)
forward_intermediate (Network _ (Theta ts) _ f) x1 = foldM (iter) x1 ts
    where 
      iter :: Vector Double 
           -> (Matrix Double, Vector Double) 
           -> Writer (DL.DList (Vector Double)) (Vector Double)
      iter x (w, b) = do 
        let z = cmap f (w #> x + b)
        writer (z, DL.singleton(z))

-- computed value -> actual value
-- backprop :: Vector Double -> Vector Double -> Network -> Theta
-- backprop y_hat y (Network _ (Theta ts) c f) = grad
--   where 
--     delta_L = grad_euclidean * cmap relu' 

main :: IO ()
main = do
  let w1 = (2 >< 2)
        [ 1.0, -0.5
        , -0.5, 1.0
        ]
      b1 = fromList [0.1, (-0.2)]
      w2 = (2 >< 2)
        [ 1.0, 1.0
        , 0.0, 1.0
        ]
      b2 = fromList [0.0, 0.3]

      theta = Theta [(w1, b1), (w2, b2)]
      dims  = Dimensions 2 2 2 2
      net   = Network dims theta euclidean relu

      xInput = fromList [1.0, -2.0]

      output = forward_intermediate net xInput

  putStrLn "Input vector:"
  print xInput

  putStrLn "Network output:"
  print output
