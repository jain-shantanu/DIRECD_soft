
@IDL_BATCH
print, 'Script started'

data = dblarr(3)
openr, lun, 'E:/PhD_Research/Task_3/input_data.txt', /get_lun
readf, lun, data
free_lun, lun

result = data * 2

openw, lun, 'E:/PhD_Research/Task_3/output_data.txt', /get_lun
printf, lun, result
free_lun, lun

print, 'Script finished'
exit
