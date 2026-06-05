{-# LANGUAGE DataKinds #-}
module Main where

import Network
import Numeric.LinearAlgebra.Static

main :: IO ()
main = do
    let w1  = matrix [1,2] :: L 2 1
        b1  = vector [0,0] :: R 2
        w2  = matrix [1,1] :: L 1 2
        b2  = vector [0]   :: R 1
        net = ConsLayer (Layer w1 b1 relu) (NilLayer (Layer w2 b2 relu))
        x   = vector [1.0] :: R 1

    let (out, bwd) = forward net x
    putStrLn $ "output (expect 3): " ++ show (extract out)

    let (deltaIn, grads) = bwd (vector [1.0] :: R 1)
    putStrLn $ "deltaInput (expect 3): " ++ show (extract deltaIn)

    case grads of
        ConsLayer (Layer gw1 gb1 _) (NilLayer (Layer gw2 gb2 _)) -> do
            putStrLn $ "grad_w1 (expect [[1],[1]]): " ++ show (extract gw1)
            putStrLn $ "grad_b1 (expect [1,1]):     " ++ show (extract gb1)
            putStrLn $ "grad_w2 (expect [[1,2]]):   " ++ show (extract gw2)
            putStrLn $ "grad_b2 (expect [1]):       " ++ show (extract gb2)
