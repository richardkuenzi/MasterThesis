from lxml import etree
import pandas as pd

import mysql.connector
from striprtf.striprtf import rtf_to_text
import re

# connect to our local MySQL instance using connection string
mydb = mysql.connector.connect(
  host="192.168.192.XXX",
  user="root",
  password="XXXX",
  database="mdx3"
)
mydb2 = mysql.connector.connect(
  host="192.168.192.XXX",
  user="root",
  password="XXXX",
  database="mdx3"
)

mycursor = mydb.cursor()  # cursor for interacting with the server
mycursor2 = mydb2.cursor()  # cursor for interacting with the server


getQuery = "SELECT id, xmlData FROM mdx3.messungen inner join sql_import on sql_import.MedicalHistoryBlockId = messungen.blockId where xmlData like '%\"tblRefraktion\"><Row><Value>AutoRef</Value>%' and refraktion_autoref is NULL \
     ;"
mycursor.execute(getQuery)  # execute SQL code like this

for row in mycursor:
    #formatted = row
    print(row[0])
    id = row[0]
    formattedrow =  ''.join(row[1]) # convert tuple to string
    #mycursor.execute(insertQuery1+formatted+insertQuery2)
    #mydb.commit()


    root = etree.fromstring(formattedrow)
    ns = {'MedicalDesktop': 'http://medicaldesktop.ch/MedicalHistoryEntrySchema.xsd'}
    counter = 0
    AutoRef = []

    def FormatAutoRef(input):
        if input != None:
            return(value.text)
        else:
            return("0")

    for table in root.findall('MedicalDesktop:Table', ns):
        attributeTableName = table.get('reference')
        #print(attributeTableName)
        
        #rows = table.find('MedicalDesktop:Row', ns)
        #print(table, "### END of TABLE")

        if attributeTableName == "tblRefraktion":
            for row in table.findall('MedicalDesktop:Row', ns):
                if row != None:
                    for value in row.findall('MedicalDesktop:Value', ns):
                        if counter == 0:
                            if value.text == "AutoRef" or value.text == "Autoref" or value.text == "autoref":
                                counter += 1
                                AutoRef.append(FormatAutoRef(value.text))
                        elif counter == 19:
                            counter = 0
                        else:
                            AutoRef.append(FormatAutoRef(value.text))
                            counter += 1
        
    insertQuery1 = "update mdx3.messungen set refraktion_autoref='AutoRef', refraktion_r_sph='"+ str(AutoRef[2]) +"'\
                    , refraktion_r_cyl='"+ str(AutoRef[3].replace("'", "")) +"'\
                    , refraktion_r_ax='"+ str(AutoRef[4].replace("'", "")) +"'\
                    , refraktion_r_add='"+ str(AutoRef[5].replace("'", "")) +"'\
                    , refraktion_r_pdpb='"+ str(AutoRef[6].replace("'", "")) +"'\
                    , refraktion_l_sph='"+ str(AutoRef[7].replace("'", "")) +"'\
                    , refraktion_l_cyl='"+ str(AutoRef[8].replace("'", "")) +"'\
                    , refraktion_l_ax='"+ str(AutoRef[9].replace("'", "")) +"'\
                    , refraktion_l_add='"+ str(AutoRef[10].replace("'", "")) +"'\
                    , refraktion_r_pdpb='"+ str(AutoRef[11].replace("'", "")) +"'\
                    , refraktion_rl_vd='"+ str(AutoRef[12].replace("'", "")) +"'\
                    , refraktion_rl_pd='"+ str(AutoRef[13].replace("'", "")) +"'\
                    , refraktion_fvisus_r='"+ str(AutoRef[14].replace("'", "")) +"'\
                    , refraktion_fvisus_l='"+ str(AutoRef[15].replace("'", "")) +"'\
                    , refraktion_nvisus_r='"+ str(AutoRef[16].replace("'", "")) +"'\
                    , refraktion_nvisus_l='"+ str(AutoRef[17].replace("'", "")) +"'\
                    , refraktion_bemerkung='"+ str(AutoRef[18].replace("'", "´")) +"'\
                    WHERE id = '"+id+"'"

    #print(insertQuery1)
    mycursor2.execute(insertQuery1)
    mydb2.commit()

    #print(AutoRef)

mydb2.close()
mydb.close()







                    











'''
table = etree.HTML(s3).find("body/table")
rows = iter(table)

headers = [col.text for col in next(rows)]
for row in rows:
    values = [col.text for col in row]
    #print(headers)
    #print(values)
    # if value exist print (output -1.25)
    #print(values[1])
    if values[0] == 'Auto Ref. Mio':
        for item in values:
            print(item)
'''