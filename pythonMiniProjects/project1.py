import random


def roll():
    roll = random.randint(1, 6)
    return roll


while True:
    players = input("Enter the number of players (2-4): ")
    if players.isdigit():
        players = int(players)
        if 2 <= players <= 4:
            break
        else:
            print("Invalid number")
    else:
        print("Read more carefully, enter again")

max_score = 50

player_scores = [0 for _ in range(players)]

while max(player_scores) < max_score:

    for player_idx in range(players):
        print("Player ", player_idx + 1, ' go!')
        current_score = 0
        while True:
            should_roll = input("would you like to roll (y) / (n)? ")
            if should_roll != "y":
                break

            value = roll()
            if value == 1:
                print("you rolled 1, turn done")
                current_score = 0
                break
            else:
                current_score += value
                print("you rolled: ", value)
            print('Your score is: ', current_score)

        player_scores[player_idx] += current_score
        print("your total score is: ", player_scores[player_idx])
