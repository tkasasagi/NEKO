import pandas as pd

dataframe = pd.read_csv('dict.csv', header = None)



dic = []

for i in range(0, 10):
    for j in range(len(dataframe[i])):
        dic.append(dataframe[i][j])
        
unicode = []
character = []
freq = []

for i in range(len(dic)):
    unicode.append(str(dic[i])[0:6])
    character.append(str(dic[i])[6:7])
    freq.append(str(dic[i])[7:])         
    
dictionary = pd.DataFrame()

dictionary['unicode'] = unicode
dictionary['character'] = character
dictionary['freq'] = freq

dictionary.to_csv("dictionary.csv", index = False)