# Python program to convert
# JSON file to CSV
 
 
import json
import csv
 
 
# Opening JSON file and loading the data
# into the variable data
with open('../datasets/g2_outputs/netpredict_overlay_traffic_json.js') as json_file:
    data = json.load(json_file)
 
traffic_data = data['traffic']
print(traffic_data[0]['bytes']) 
# now we will open a file for writing
data_file = open('netpredict_overlay_traffic.csv', 'w')
 
# create the csv writer object
csv_writer = csv.writer(data_file)
 
# Counter variable used for writing
# headers to the CSV file
count = 0
 
for traf in traffic_data:
    if count == 0:
 
        # Writing headers of CSV file
        header = traf.keys()
        csv_writer.writerow(header)
        count += 1
 
    # Writing data of CSV file
   # print(traf.values())
    csv_writer.writerow(traf.values())
 
data_file.close()