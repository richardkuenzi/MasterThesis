from striprtf.striprtf import rtf_to_text
import re

def cleanxml():
    # Input raw data from medicaldesktop
    # ouptut xml/rtf cleaned data
    text = re.sub('<[^<]+>', "",r'<Entry xmlns="http://medicaldesktop.ch/MedicalHistoryEntrySchema.xsd" Type="Anamnese und Befund"><Richbox reference="txtMigration">{\rtf1\fbidis\ansi\ansicpg1252\deff0\deflang1031{\fonttbl{\f0\fnil\fcharset0 Microsoft Sans Serif;}} Lorem ipsum\par } </Richbox></Entry>')
    text = " ".join(text.split())
    text = text.replace(";", "")

    text = rtf_to_text(text)

    text = " ".join(text.splitlines())

    print(text)

cleanxml()