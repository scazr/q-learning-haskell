{-# LANGUAGE TupleSections #-}

import Data.List
import Numeric.Natural
import System.Random
import Data.Function (on)
import Control.Applicative
import qualified Data.Map.Strict as M
import Data.Maybe
import Control.Monad (foldM, when, replicateM)

-- Matrix functions

newtype Matrix a = Matrix [[a]] deriving (Eq, Ord)
type MatrixCoords = (Natural, Natural)

instance Show a => Show (Matrix a) where
    show (Matrix m) =
        intercalate "\n" $ ["["] ++ ["  [" ++ unwords (map show row) ++ "]" | row <- m] ++ ["]"]

matrixZeros :: Natural -> Natural -> Matrix Double
matrixZeros i j = Matrix (replicate (fromIntegral i) $ replicate (fromIntegral j) 0)

matrixFull :: Natural -> Natural -> a -> Matrix a
matrixFull i j x = Matrix (replicate (fromIntegral i) $ replicate (fromIntegral j) x)

matrixNDenumerate :: Matrix a -> [(MatrixCoords, a)]
matrixNDenumerate (Matrix m) = [((i, j), x) | (i, row) <- zip [0..] m, (j, x) <- zip [0..] row]

matrixMaximum :: Ord a => Matrix a -> (MatrixCoords, a)
matrixMaximum m = maximumBy (compare `on` snd) $ matrixNDenumerate m

matrixDiagonal :: Matrix a -> [a]
matrixDiagonal (Matrix m) = [row !! i | (row, i) <- zip m [0..], i < length row]

matrixAntiDiagonal :: Matrix a -> [a]
matrixAntiDiagonal (Matrix m) = [row !! j | (row, i) <- zip m [0..], let j = (length row - 1) - i, j >= 0]

unMatrix :: Matrix a -> [[a]]
unMatrix (Matrix m) = m

setCell :: Matrix a -> MatrixCoords -> a -> Matrix a
setCell (Matrix rows) (r,c) x =
    Matrix $ take ri rows ++ [newRow] ++ drop (ri+1) rows
  where
    ri = fromIntegral r
    ci = fromIntegral c
    row = rows !! ri
    newRow = take ci row ++ [x] ++ drop (ci+1) row

getCell :: Matrix a -> MatrixCoords -> a
getCell (Matrix rows) (r,c) = (rows !! fromIntegral r) !! fromIntegral c

-- TicTacToe specifics

data TicTacToe = TTTEmpty | TTTO | TTTX deriving (Show, Eq, Ord)
data TicTacToeRewards = Win | Draw | Loss deriving (Show, Eq)

board :: Matrix TicTacToe
board = matrixFull 3 3 TTTEmpty

matchPatterns :: Matrix TicTacToe -> [[TicTacToe]]
matchPatterns (Matrix m) =
    let rows = m
        cols = transpose m
        diag = matrixDiagonal (Matrix m)
        adiag = matrixAntiDiagonal (Matrix m)
    in rows ++ cols ++ [diag, adiag]

checkSegment :: [TicTacToe] -> Maybe TicTacToe
checkSegment [] = Nothing
checkSegment (x:xs)
    | x == TTTEmpty = Nothing
    | all (== x) xs = Just x
    | otherwise = Nothing

checkWinner :: Matrix TicTacToe -> Maybe TicTacToe
checkWinner m = foldr ((<|>) . checkSegment) Nothing (matchPatterns m)

isBoardFull :: Matrix TicTacToe -> Bool
isBoardFull (Matrix m) = all (/= TTTEmpty) (concat m)

isTerminal :: Matrix TicTacToe -> Bool
isTerminal s = isJust (checkWinner s) || isBoardFull s

availableActions :: Matrix TicTacToe -> [MatrixCoords]
availableActions m = [coord | (coord, cell) <- matrixNDenumerate m, cell == TTTEmpty]

rewardForState :: Matrix TicTacToe -> TicTacToeRewards
rewardForState s = case checkWinner s of
    Just TTTX -> Win
    Just TTTO -> Loss
    _ | isBoardFull s -> Draw
      | otherwise -> Draw

-- Q-Learning and Hyperparameters
type State = Matrix TicTacToe
type Action = MatrixCoords
type QTable = M.Map (State, Action) Double

learningRate :: Double
learningRate = 0.2

discountFactor :: Double
discountFactor = 0.9

explorationProbability :: Double
explorationProbability = 0.1

getQ :: QTable -> State -> Action -> Double
getQ q s a = M.findWithDefault 0 (s,a) q

maxQ :: QTable -> State -> Double
maxQ q s =
    case availableActions s of
        [] -> 0
        acts -> maximum $ map (getQ q s) acts

chooseAction :: StdGen -> QTable -> State -> (Action, StdGen)
chooseAction gen q s =
    let acts = availableActions s
    in if null acts
       then (error "No actions available", gen)
       else
        let (r, gen') = randomR (0.0, 1.0) gen :: (Double, StdGen)
        in if r < explorationProbability
           then let (idx, gen'') = randomR (0, length acts - 1) gen' in (acts !! idx, gen'')
           else 
               let qs = map (\a -> (getQ q s a, a)) acts
                   bestVal = maximum (map fst qs)
                   bestActs = [a | (v,a) <- qs, v == bestVal]
                   (idx, gen'') = randomR (0, length bestActs - 1) gen'
               in (bestActs !! idx, gen'')

updateQ :: QTable -> State -> Action -> Double -> State -> QTable
updateQ q s a r s' =
    let old = getQ q s a
        target = r + discountFactor * maxQ q s'
        newQ = old + learningRate * (target - old)
    in M.insert (s,a) newQ q

chooseRandomAction :: StdGen -> State -> (Action, StdGen)
chooseRandomAction gen s =
    let acts = availableActions s
    in if null acts then (error "No actions for opponent", gen)
       else let (idx, g') = randomR (0, length acts - 1) gen in (acts !! idx, g')

applyAction :: State -> Action -> TicTacToe -> State
applyAction s a player = setCell s a player

simulateEpisode :: StdGen -> QTable -> (QTable, StdGen)
simulateEpisode gen q0 = go gen q0 board True Nothing
  where
    go :: StdGen -> QTable -> State -> Bool -> Maybe (State, Action) -> (QTable, StdGen)
    go g q s agentTurn lastAgentMove
      | isTerminal s =
          case lastAgentMove of
            Nothing -> (q, g)
            Just (sPrev, aPrev) ->
              let r  = rewardToDouble (rewardForState s)
                  q' = updateQ q sPrev aPrev r s
              in (q', g)

      -- X turn
      | agentTurn =
          let (a, g1) = chooseAction g q s
              s'      = applyAction s a TTTX
              r       = if isTerminal s'
                          then rewardToDouble (rewardForState s')
                          else 0
              q'      = updateQ q s a r s'
              last'   = Just (s, a)
          in go g1 q' s' False last'

      -- O turn
      | otherwise =
          let (a, g1) = chooseRandomAction g s
              s'      = applyAction s a TTTO
          in go g1 q s' True lastAgentMove

rewardToDouble :: TicTacToeRewards -> Double
rewardToDouble Win  = 1.0
rewardToDouble Draw = 0.0
rewardToDouble Loss = -1.0

-- Training loop
train :: StdGen -> Int -> QTable -> IO QTable
train gen 0 q = return q
train gen n q = do
    let (q', gen') = simulateEpisode gen q
    when (n `mod` 5000 == 0) $ putStrLn $ "Episodes remaining: " ++ show n
    train gen' (n-1) q'

bestActionIO :: QTable -> State -> IO (Maybe Action)
bestActionIO q s =
  case availableActions s of
    []   -> return Nothing
    acts -> do
      let qs       = map (\a -> (getQ q s a, a)) acts
          bestVal  = maximum (map fst qs)
          bestActs = [a | (v,a) <- qs, v == bestVal]
      idx <- randomRIO (0, length bestActs - 1)
      return (Just (bestActs !! idx))

playWithAgent :: QTable -> IO ()
playWithAgent q = playLoop board where
  playLoop s = do
    putStrLn "\nCurrent board:"
    print s
    if isTerminal s
      then putStrLn $ "Game over: " ++ show (rewardForState s)
      else do
        putStrLn "Your move (format: row col) where 0<=row,col<=2, you're O."
        line <- getLine
        case words line of
          [rs, cs] ->
            let r = read rs :: Natural
                c = read cs :: Natural
                coord = (r,c)
            in if coord `elem` availableActions s
                then do
                  let s' = applyAction s coord TTTO
                  if isTerminal s' then do
                      print s'
                      putStrLn $ "Game over: " ++ show (rewardForState s')
                    else do
                      ma <- bestActionIO q s'
                      case ma of
                        Nothing -> putStrLn "Agent has no moves. (draw?)"
                        Just a -> do
                          let s'' = applyAction s' a TTTX
                          putStrLn $ "Agent plays: " ++ show a
                          if isTerminal s'' then do
                              print s''
                              putStrLn $ "Game over: " ++ show (rewardForState s'')
                            else playLoop s''
                else putStrLn "Invalid move (occupied or out of range)" >> playLoop s
          _ -> putStrLn "Bad input" >> playLoop s

-- Main
main :: IO ()
main = do
    gen <- getStdGen
    putStrLn "Training Q-Learning agent for Tic-Tac-Toe..."
    qTrained <- train gen 30000 M.empty
    putStrLn "Training complete."
    playWithAgent qTrained