import mmap
from try_gray import filterText
from match_word import findM

text =", 3 4 2 a v d 3 92834"

idd = text.split()
rs = max(idd,key=len)
print(rs)

