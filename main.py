import csv
from io import open
import matplotlib.pyplot as plt



variables = []

with open('X_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        line = [row[0], row[1], row[2], row[3], row[4]]
        #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
        variables.append(line)
        line_count += 1
    #print(f'Processed {line_count} lines.')
    print(variables)
