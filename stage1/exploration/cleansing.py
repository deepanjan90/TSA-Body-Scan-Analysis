import csv
from os import listdir
import shutil

aps_files = set(listdir('./aps/'))
not_present = []
present = {}
count = 0
total_row = 0

with open('stage1_labels.csv', 'r') as csvfile:
	filereader = csv.reader(csvfile)
	next(filereader)
	for row in filereader:
		row = row[0]
		total_row = total_row + 1
		row = row.split('_')
		filename = row[0] + '.aps'
		present[filename] = 1 + (present[filename] if filename in present else 0)

csv_files = set(present.keys())
missing_set = (csv_files - aps_files) | (aps_files - csv_files)

print('--------------------------')
print('Missing Set')
print('----')
for missing in missing_set:
	print(missing)
	shutil.move(missing, '../missing_labels_aps/')

print('--------------------------')
print('Excess Zones')
print('----')
for key in present:
	value = present[key]
	if value != 17:
		print(key + " Zone count: " + str(value))

print('--------------------------')
print('CSV row count: ' + str(total_row))
print('Total files: ' + str(len(aps_files)))
print('unaccounted file sets: ' + str(len(missing_set)))

