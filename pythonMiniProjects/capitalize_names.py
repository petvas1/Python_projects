names = ['bro code', 'dino pes velky', 'maly', 'muro-sdf 3']
capitalized = []
for name in names:
    name_split = name.split()
    name_capitalized = ''

    for part in name_split:
        if part[0].islower():
            part = part.capitalize()
        name_capitalized += part + ' '

    capitalized.append(name_capitalized[:-1])

print(capitalized)

# names = ['bro code', 'dino pes velky', 'maly', 'muro-sdf 3']
# capitalized = []
# for name in names:
#     name_split = name.split()
#     name_capitalized = []
#
#     for part in name_split:
#         if part[0].islower():
#             part = part.capitalize()
#         name_capitalized.append(part)
#
#     new_name = ' '.join(name_capitalized)
#     capitalized.append(new_name)
#
# print(capitalized)
