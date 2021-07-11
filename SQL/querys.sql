SELECT tblActivityRecording.code as code, tblActivityRecording.id as ActivityRecordingId, tblActivityRecording.description, tblActivityRecording.serviceSheetId as ConsultationId, tblMedicalHistoryBlock.id as MedicalHistoryBlockId,CONVERT(CHAR(19),tblActivityRecording.created,4) as ActivityRecordingCreated, CONVERT(CHAR(19),tblMedicalHistoryBlock.created,4) as MedicalHistoryBlockCreated, CONVERT(CHAR(8000), tblMedicalHistoryEntry.xmlData, 1) as MedicalHistoryEntry, tblMedicalHistoryEntry.id as MedicalHistoryEntryId
  FROM [MDX3].[dbo].[tblCase]
  INNER JOIN tblConsultation ON tblCase.id=tblConsultation.serviceGroupId
  INNER JOIN tblActivityRecording ON tblConsultation.id=tblActivityRecording.serviceSheetId
  INNER JOIN tblMedicalHistoryBlock ON CONVERT(CHAR(19),tblActivityRecording.created,4)=CONVERT(CHAR(19),tblMedicalHistoryBlock.created,4)
  INNER JOIN tblMedicalHistory ON tblMedicalHistoryBlock.medicalHistoryId=tblMedicalHistory.id
  INNER JOIN tblMedicalHistoryEntry ON tblMedicalHistoryBlock.id=tblMedicalHistoryEntry.blockId
  INNER JOIN tblPatient ON tblMedicalHistory.patientId=tblPatient.id and tblCase.patientId=tblPatient.id
  where
  tblActivityRecording.created > '2018-01-01 00:00' and tblActivityRecording.deleted = 0 and tblMedicalHistoryBlock.deleted = 0 and tblMedicalHistory.deleted = 0 and tblConsultation.deleted = 0 and tblPatient.deleted = 0 and tblMedicalHistoryEntry.deleted = 0
  and CONVERT(NVARCHAR(MAX), tblMedicalHistoryEntry.xmlData, 1) like '%Anamnese und Befund%'

  order by tblMedicalHistoryBlock.id