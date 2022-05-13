import re


if __name__ == '__main__':

    f = open('MSU-MFSD.csv', 'r')
    new_csv = open('fine_grain/MSU-REAL.csv', 'w')
    pattern = '(.+)/print(.+).zip,(.+)'


    new_list = []
    for line in f.readlines():
        if re.match(pattern,line):
            print(line)
            new_list.append(line)


    new_csv.writelines(new_list)
    f.close()
    new_csv.close()

