import joblib
import numpy as np


# Load the saved model
models = [
    ('linear_svm_model_single.pkl', 'Easy'),
    ('knn_model_single.pk1', 'Medium'),
    ('mlp_model_single.pk1', 'Hard'),
    ('knn_model_multi.pk1', 'Easy'),
    ('lr_model_multi.pk1', 'Medium'),
    ('mlp_model_multi.pk1', 'Hard')
]

loaded_model = joblib.load(models[0][0])

board = [0 for _ in range(9)]
difficulty = 0


def get_symbol(val: int) -> str:
    if val == 0:
        return ' '
    elif val == 1:
        return 'X'
    else:
        return 'O'


def displayBoard():
    print(f"Difficulty: {models[difficulty][1]}")
    print('--------------')
    print('   |'.join(get_symbol(x) for x in board[0:3]))
    print('--------------')
    print('   |'.join(get_symbol(x) for x in board[3:6]))
    print('--------------')
    print('   |'.join(get_symbol(x) for x in board[6:9]))
    print('--------------')


def get_player_move() -> int:
    displayBoard()
    res = -1
    while True:
        try:
            res = int(input("Choose a space [1-9]: ")) - 1
            if res >= 0 and res <= 8:
                if board[res] == 0:
                    break
                print("Must pick empty space")
            else:
                print("Must choose between 1-9")
        except:
            print("Must choose between 1-9")

    return res


def get_random() -> int:
    for i in range(9):
        if board[i] == 0:
            return i
    print('ERROR: No empty spaces and game not ended')
    exit(1)


def handle_prediction(prediction: np.ndarray) -> int:
    if len(prediction) > 1:
        # Classification Model
        return int(prediction[0])
    else:
        # Regressor Model
        ind = -1
        min_diff = 10
        # Get the closest position to 1
        for i, num in enumerate(prediction[0]):
            diff = 1 - abs(num)
            if diff < min_diff:
                min_diff = diff
                ind = i

        if ind == -1:
            print(f"Regressor model failed to make decision")
            return get_random()
        else:
            return ind


def get_ai_move() -> int:
    input_data = np.array(board).reshape(1, -1) # Format input
    prediction = loaded_model.predict(input_data)   # Predict
    pos = handle_prediction(prediction)

    # Sanitation check
    if pos < 0 or pos > 8:
        print(f"AI picked a number outside 1-9: {pos+1}")
        pos = get_random()
    if board[pos] != 0:
        print(f'AI chose a non empty space: {pos+1}')
        pos = get_random()

    return pos
    

def game_ended() -> int:
    # Define winning combinations (indices of the board)
    winning_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)              # Diagonals
    ]

    # Check each winning combination
    for combo in winning_combinations:
        a, b, c = combo
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]  # Return the winning player (1 or -1)
    
    return 0 if 0 in board else 2


def init_game():
    # Game started
    turn = 1
    winner = 0
    while winner == 0:
        if turn == 1:
            pos = get_player_move()
            board[pos] = 1
            turn = 2
        elif turn == 2:
            # print(board)
            pos = get_ai_move()
            board[pos] = -1
            turn = 1

        winner = game_ended()
    
    # Game Ended
    if winner == 2:
        print("\n\nIt was a tie! Good job to both players :)")
    else:
        print(f"\n\nCongrats player {'X' if winner == 1 else 'O'}! You won :)")
    displayBoard()


def set_difficulty():
    global loaded_model, difficulty
    print("Classifier models:")
    print(f"Easy (1)")
    print(f"Medium (2)")
    print(f"Hard (3)\n")

    print("Regressor Models:")
    print(f"Easy (4)")
    print(f"Medium (5)")
    print(f"Hard (6)\n")

    while True:
        try:
            res = int(input("Select your difficulty level [1-6]: "))
            if res >= 1 and res <= 6:
                difficulty = res - 1
                loaded_model = joblib.load(models[difficulty][0])
                break
            print("Invalid")
        except:
            print("Invalid")


def main():
    set_difficulty()
    init_game()
    # get_ai_move()


if __name__ == '__main__':
    main()
