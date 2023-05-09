import mmap
import time


def Lower(s):
    if type(s) == type(u""):
        return s.lower()
    return str(s,"utf8").lower().encode("utf8")

def findM(s):
    tmp = s
    file_path = '.\general_dict.txt'
    rsd = ["họ", "tên", "sinh","ngày", "nguyên", "quán", "nơi", "thường","trú", "số"]
    tmp = Lower(tmp)
    if tmp in rsd: # delete sub title
        return False
    if tmp.isdigit(): # accept all number
        return True
    with open(file_path, 'r',encoding='utf-8') as file:
        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_file:
            if mmap_file.find(tmp.encode('utf-8')) != -1:
                return True
            return False

if __name__ == "__main__":
    now = time.time()
    search_string = 'Hown'
    if findM(search_string):
        print(f'found {search_string}') 
    else:
        print("not found")
    print(f'Time exercuted {(time.time() - now):.2f}')
