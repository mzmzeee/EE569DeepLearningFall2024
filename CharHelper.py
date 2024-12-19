import random
import time
import math
from io import open
import unicodedata


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s,all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename,all_letters):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip(),all_letters) for line in some_file]

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

