import csv
import pandas as pd

img_id = []
zone_no = []
prob = []

with open('stage1_labels.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if(row[0]!='Id'):
            a = row[0].split('_')
            img_id.append(a[0])
            zone_no.append(int(a[-1].split('Zone')[-1]))
            prob.append(int(row[1]))

pandas_combined_list = pd.DataFrame({
    'Id':img_id,
    'Zone':zone_no,
    'Prob':prob
    })

idList = []
probList = []

pandas_grouped_list = pandas_combined_list.sort_values(['Zone'],ascending=True).groupby(['Id'])

for key, item in pandas_grouped_list:
    item = pandas_grouped_list.get_group(key)
    idList.append(item['Id'].tolist()[0])
    probList.append(item['Prob'].tolist())

print(probList)




