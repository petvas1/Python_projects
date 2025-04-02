with open('text_r_w.txt', 'r+') as f:
    # f_contents = f.readlines()
    # for line in f_contents:
    #     print(line.strip())
    lines = [line.strip() for line in f]
    for line in lines:
        print(line)
    lines2 = ' '.join(lines)
    lines2 = lines2.replace('"', '')
    lines2 = lines2.replace('. ', '.\n')
    # f_contents = f.read(100)
    # f_contents = f_contents.replace('"', '')
    # f_contents = f_contents.replace('. ', '.\n')
    f.seek(0)
    f.write(lines2)
