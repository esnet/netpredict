## First login to mysql and run
## create database Netpredict
## and then run this script after uncommenting line 27

import mysql.connector

def create_database():
  mydb = mysql.connector.connect(
    host="localhost",
    user="netpred",
    password="rootroot",
    database="netpredictdb"
  )

  cursor=mydb.cursor()

  # cursor.execute("create database NetPredict")
  cursor.execute('create table predicted_values(duration VARCHAR(50), predict DOUBLE);')
  graph = ['14 may', '15 may', '16 may', '17 may', '18 may', '19 may', '20 may', '21 may']
  predict = [0.11, 0.13, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6]
  for (duration,predicted) in zip(graph,predict):
    cursor.execute(f'insert into predicted_values (duration,predict) values ("{duration}",{predicted});')
  else:
    mydb.commit()
  cursor.close()

#create_database()

def read_database():
  mydb = mysql.connector.connect(
    host="localhost",
    user="netpred",
    password="rootroot",
    database='netpredictdb'
  )

  cursor=mydb.cursor()

  cursor.execute("select * from predicted_values;")
  data=cursor.fetchall()
  graph,predict=[i[0] for i in data],[i[1] for i in data]

  return [graph,predict]
