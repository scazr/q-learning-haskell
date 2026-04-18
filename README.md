# Q-Learning in Haskell

Q-Learning implementations in Haskell, featuring a custom-built matrix library (**NumHask**) used as the backbone for state representation.

## Structure

```
accepted/
├── gridworld.hs     # Q-Learning on a 3x3 GridWorld
└── tictactoe.hs     # Q-Learning agent for Tic-Tac-Toe
```

---

## NumHask — Custom Matrix Library

Both implementations share a self-contained matrix module built on top of Haskell's native lists. It provides the `Matrix a` type used to represent states throughout the Q-Learning agents.

### Types

| Type | Definition | Description |
|---|---|---|
| `Matrix a` | `newtype Matrix [[a]]` | A 2D grid of any type `a` |
| `MatrixCoords` | `(Natural, Natural)` | A `(row, col)` coordinate pair |

### Functions

| Function | Signature | Description |
|---|---|---|
| `matrixZeros` | `Natural -> Natural -> Matrix Double` | Matrix filled with `0.0` |
| `matrixFull` | `Natural -> Natural -> a -> Matrix a` | Matrix filled with a constant value |
| `matrixNDenumerate` | `Matrix a -> [(MatrixCoords, a)]` | All cells paired with their coordinates |
| `matrixMaximum` | `Ord a => Matrix a -> (MatrixCoords, a)` | Cell with the maximum value |
| `matrixDiagonal` | `Matrix a -> [a]` | Main diagonal (top-left → bottom-right) |
| `matrixAntiDiagonal` | `Matrix a -> [a]` | Anti-diagonal (top-right → bottom-left) |
| `unMatrix` | `Matrix a -> [[a]]` | Unwrap to a nested list |
| `getCell` | `Matrix a -> MatrixCoords -> a` | Read one element |
| `setCell` | `Matrix a -> MatrixCoords -> a -> Matrix a` | Return a new matrix with one cell changed |

---

## GridWorld

A Q-Learning agent navigating a **3×3 grid** from position `(0,0)` to the terminal state `(2,2)`.

### State & Action Space

- **State:** A `3×3 Matrix Double` with a single `1.0` marking the agent's position — all other cells are `0.0`
- **Actions:** `AUp`, `ADown`, `ALeft`, `ARight`
- **Reward:** `+1.0` on reaching `(2,2)`, `0.0` otherwise

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (α) | `0.1` |
| Discount factor (γ) | `0.95` |
| Exploration rate (ε) | `0.2` |
| Episodes | `1000` |

### Q-Table

The Q-table is a `Map (State, Action) Double`. Values default to `0` for unseen state-action pairs. After training, the program prints the learned Q-values and the greedy policy for every non-terminal state.

### Running

```bash
runghc gridworld.hs
```

Sample output:
```
Treinando Q-learning no grid 3x3...
Q(s,a) aprendido:
Estado (0,0) | AUp: 0.0 | ADown: 0.51 | ALeft: 0.0 | ARight: 0.49
...
Política final (greedy):
Estado (0,0) -> ADown
...
```

---

## Tic-Tac-Toe

A Q-Learning agent that learns to play **Tic-Tac-Toe** as `X` against a random opponent (`O`), then lets you play against it interactively.

### State & Action Space

- **State:** A `3×3 Matrix TicTacToe` where each cell is `TTTEmpty`, `TTTX`, or `TTTO`
- **Actions:** Any `MatrixCoords` pointing to an empty cell
- **Rewards:** `+1.0` (Win), `0.0` (Draw), `-1.0` (Loss)

### Data Types

```haskell
data TicTacToe        = TTTEmpty | TTTO | TTTX
data TicTacToeRewards = Win | Draw | Loss
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | `0.2` |
| Discount factor | `0.9` |
| Exploration rate (ε) | `0.1` |
| Episodes | `30000` |

### Win Detection

The `checkWinner` function checks all rows, columns, the main diagonal, and the anti-diagonal using `matchPatterns`, `matrixDiagonal`, and `matrixAntiDiagonal` from NumHask.

### Running

```bash
runghc tictactoe.hs
```

After training, you play as `O` and input moves as `row col` (0-indexed):

```
Training Q-Learning agent for Tic-Tac-Toe...
Episodes remaining: 30000
...
Training complete.

Current board:
[
  [TTTEmpty TTTEmpty TTTEmpty]
  [TTTEmpty TTTEmpty TTTEmpty]
  [TTTEmpty TTTEmpty TTTEmpty]
]
Your move (format: row col) where 0<=row,col<=2, you're O.
```

---

## Dependencies

- [`random`](https://hackage.haskell.org/package/random) — for ε-greedy exploration
- [`containers`](https://hackage.haskell.org/package/containers) — for `Data.Map.Strict` (Q-table)
- `Numeric.Natural` — for non-negative matrix coordinates (included in `base`)

Install with Cabal or Stack:

```bash
cabal install random containers
```
