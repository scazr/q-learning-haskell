{-# LANGUAGE TupleSections #-}

import System.Random (randomRIO)
import qualified Data.Map.Strict as M
import Data.List (maximumBy)
import Data.Ord (comparing)
import Numeric.Natural

-- Matrix functions

newtype Matrix a = Matrix [[a]] deriving (Eq, Ord)
type MatrixCoords = (Natural, Natural)

instance Show a => Show (Matrix a) where
    show (Matrix m) =
        unlines $ ["["] ++ ["  " ++ show row | row <- m] ++ ["]"]

matrixZeros :: Natural -> Natural -> Matrix Double
matrixZeros i j = Matrix (replicate (fromIntegral i) (replicate (fromIntegral j) 0))

unMatrix :: Matrix a -> [[a]]
unMatrix (Matrix m) = m

setCell :: Matrix a -> MatrixCoords -> a -> Matrix a
setCell (Matrix rows) (r,c) x =
    Matrix (take ri rows ++ [newRow] ++ drop (ri+1) rows)
  where
    ri = fromIntegral r
    ci = fromIntegral c
    row = rows !! ri
    newRow = take ci row ++ [x] ++ drop (ci+1) row

getCell :: Matrix a -> MatrixCoords -> a
getCell (Matrix rows) (r,c) = (rows !! fromIntegral r) !! fromIntegral c


-- Q-learning GridWorld

type Pos = (Int, Int)
type State = Matrix Double

data Action = AUp | ADown | ALeft | ARight
  deriving (Eq, Ord, Show, Enum, Bounded)

allActions :: [Action]
allActions = [minBound .. maxBound]

type QTable = M.Map (State, Action) Double

gridSize :: Int
gridSize = 3  -- 3x3

gridMax :: Int
gridMax = gridSize - 1

-- Hiperparameters

alpha, gamma, epsilon :: Double
alpha   = 0.1
gamma   = 0.95
epsilon = 0.2

numEpisodes :: Int
numEpisodes = 1000

terminalPos :: Pos
terminalPos = (2, 2)

stateFromPos :: Pos -> State
stateFromPos (x,y) =
    setCell
        (matrixZeros (fromIntegral gridSize) (fromIntegral gridSize))
        (fromIntegral x, fromIntegral y)
        1.0

posFromState :: State -> Pos
posFromState (Matrix rows) =
    head [ (i,j)
         | (i,row) <- zip [0..] rows
         , (j,v)   <- zip [0..] row
         , v == 1.0
         ]

step :: Pos -> Action -> (Pos, Double, Bool)
step (x, y) a =
  let (nx, ny) = case a of
        AUp    -> (max 0 (x - 1), y)
        ADown  -> (min gridMax (x + 1), y)
        ALeft  -> (x, max 0 (y - 1))
        ARight -> (x, min gridMax (y + 1))
      s' = (nx, ny)
      reward = if s' == terminalPos then 1.0 else 0.0
      done   = s' == terminalPos
  in (s', reward, done)

getQ :: QTable -> State -> Action -> Double
getQ q s a = M.findWithDefault 0 (s, a) q

setQ :: QTable -> State -> Action -> Double -> QTable
setQ q s a v = M.insert (s, a) v q

bestAction :: QTable -> State -> Action
bestAction q s =
    fst $ maximumBy (comparing snd)
        [ (a, getQ q s a) | a <- allActions ]

maxQNext :: QTable -> State -> Double
maxQNext q s =
    maximum [ getQ q s a | a <- allActions ]

chooseAction :: QTable -> State -> IO Action
chooseAction q s = do
    r <- randomRIO (0.0, 1.0 :: Double)
    if r < epsilon
       then randomAction
       else return (bestAction q s)

randomAction :: IO Action
randomAction = do
    let n = length allActions
    i <- randomRIO (0, n - 1)
    return (allActions !! i)

runEpisode q0 = loop (0,0) q0
  where
    loop :: Pos -> QTable -> IO QTable
    loop pos q
      | pos == terminalPos = return q
      | otherwise = do
          let s = stateFromPos pos
          a <- chooseAction q s
          let (pos', r, done) = step pos a
              s' = stateFromPos pos'
              oldQ = getQ q s a
              maxNext = if done then 0 else maxQNext q s'
              newQ = oldQ + alpha * (r + gamma * maxNext - oldQ)
              q' = setQ q s a newQ
          if done
            then return q'
            else loop pos' q'

-- Training

train :: Int -> QTable -> IO QTable
train 0 q = return q
train n q = do
    q' <- runEpisode q
    train (n - 1) q'

-- IO 

printQTable :: QTable -> IO ()
printQTable q = do
  putStrLn "Q(s,a) aprendido:"
  mapM_ printState [ (i,j) | i <- [0..gridMax], j <- [0..gridMax] ]
  where
    printState pos = do
      let s = stateFromPos pos
          qUp    = getQ q s AUp
          qDown  = getQ q s ADown
          qLeft  = getQ q s ALeft
          qRight = getQ q s ARight
      putStrLn $ "Estado " ++ show pos ++
                 " | AUp: "    ++ show qUp    ++
                 " | ADown: "  ++ show qDown  ++
                 " | ALeft: "  ++ show qLeft  ++
                 " | ARight: " ++ show qRight

printGreedyPolicy :: QTable -> IO ()
printGreedyPolicy q = do
  putStrLn "\nPolítica final (greedy):"
  mapM_ printState [ pos | pos <- allPositions, pos /= terminalPos ]
  where
    allPositions = [ (i,j) | i <- [0..gridMax], j <- [0..gridMax] ]
    printState pos =
      putStrLn $ "Estado " ++ show pos ++
                 " -> " ++ show (bestAction q (stateFromPos pos))

-- Main 

main :: IO ()
main = do
  putStrLn "Treinando Q-learning no grid 3x3..."
  qFinal <- train numEpisodes M.empty
  printQTable qFinal
  printGreedyPolicy qFinal