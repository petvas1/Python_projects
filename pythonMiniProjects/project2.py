with open('text2.txt', 'r') as f:
    story = f.read()

words = []          # can use set() instead, append -> add
start_of_word = -1
target_start = '<'
target_end = '>'

for i, char in enumerate(story):
    if char == target_start:
        start_of_word = i

    if char == target_end and start_of_word != -1:
        word = story[start_of_word: i + 1]
        words.append(word)
        start_of_word = -1

words = list(dict.fromkeys(words))

answers = {}
for word in words:
    answer = input('Enter a word for ' + word + ': ')
    answers[word] = answer

for word in words:
    story = story.replace(word, answers[word])

print(story)
