script_path = FILE_DIRNAME(ROUTINE_FILEPATH())
parts = STRSPLIT(script_path, '/\', /EXTRACT)  
parent_dir = STRJOIN(parts[0:N_ELEMENTS(parts)-2], '/') + '/Events/'     
CD, parent_dir
script_path = FILE_DIRNAME(ROUTINE_FILEPATH())
print, 'Script Directory: ', script_path +''
end