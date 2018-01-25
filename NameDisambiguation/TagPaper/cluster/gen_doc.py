__author__ = 'ssm'
#Given a path, each sub pash of it means a level-1 cluster.
#Each file in a sub path, is a level-2 cluster.
#In each file, contains the keywords in this cluster.
#The first line is the avarage distance
#This program choose the first 5 keywork in the level-2 cluster.
import os
import codecs

path = "./skm_iter_50_2nd/"

#Input the list of keywords, strip the space, than make up a new string with them.
#In the formated string, each word is seperated with space, end with ';'
#@param keywords: the list of keywords, each element of the list is a keyword.
def keywords_formation(keywords):
    formated_str = ''
    for word in keywords:
        formated_str += word.strip() + ' '
    formated_str += ';'
    return formated_str

#Input the first line of the clustring result file, delete the useless char of it.
#@param first_line: the first line of the result file.
def first_line_formation(first_line):
    first_line = first_line.replace('[', '')
    first_line = first_line.replace(']', '')
    return first_line.strip()

dirs = os.listdir(path)
result_str = ''
level1_cluster_index = 0
for dir in dirs:
    files = os.listdir(path + dir)
    result_str += str(level1_cluster_index) + '\n'
    level1_cluster_index += 1
    for file in files:
        if file.endswith(".txt"):
            with codecs.open((path + dir + "/" + file),'rb','utf-8') as f:
                lines = f.readlines()
                result_str += '\t-\n'
                #st += '\t\t- ' + str(len(lines)) + ' ' + tostring1(lines[0]) + ' ' + tostring(lines[1:6]) + '\n'
                result_str += '\t\t- ' + first_line_formation(lines[0]) + ' ' + keywords_formation(lines[1:6]) + '\n'

with codecs.open('doc.txt', 'wb', 'utf-8') as f:
    f.write(result_str)
