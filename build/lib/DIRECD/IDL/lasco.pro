;step = 1  ;;; get LASCO data
;step = 2 ;;; calibrate LASCO data
;step = 3 ;;; make base difference data
;step = 4 ;;; save base difference to fits 
event = FILE_DIRNAME(ROUTINE_FILEPATH())
step =4
IF step EQ 1 then BEGIN
;-- Download LASCO C2/C3 data from VSO
   tstart='01-Aug-2015 07:00:00';'13-Mar-2012 17:00:00'
   tend='01-Aug-2015 07:30:00';'13-Mar-2012 19:00:00'
   a1c2=vso_search(tstart, tend, instrument='LASCO', detector='C2')
   FILE_MKDIR, event + '/LASCO/orig/'
   a2c2=vso_get(a1c2, out_dir= event + '/LASCO/orig/')
ENDIF
step =4
IF step EQ 2 then BEGIN
;-- process LASCO level 0.5 data:
  lvl_05_list=file_list(event + '/LASCO/orig/', '*.fts')
  mreadfits, lvl_05_list, hdr_05
  lvl_05_num=n_elements(lvl_05_list)
  FILE_MKDIR, event + '/LASCO/cal/'
  for i=0,lvl_05_num-1 do begin
    print,'+++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print, ' '
    print, 'processing '+strtrim(i,1)+'/'+strtrim(lvl_05_num-1,1)+'...'
    print, ' '
    print,'+++++++++++++++++++++++++++++++++++++++++++++++++++++'
    
    reduce_level_1, lvl_05_list[i], /NO_CALFAC, savedir=event + '/LASCO/cal/'
  endfor
ENDIF


step =4
IF step EQ 3 then BEGIN

   path_data= event + '/LASCO/cal/' ;;; where calibrated data are saved
   FILE_MKDIR, event + '/LASCO/base_diff_sav/'
  path_save=event + '/LASCO/base_diff_sav/' ;;; where to save base difference data

   files = findfile(path_data +'*.fts' ) ;;;name of files in the folder
   N=n_elements(files)                   ;;;number of files


  fits2map, files, maps    ;;;to read in idl and create maps (actual image data)

 
  time_start=maps[0].time  ;;;base image (first image in the array, but it can be any needed image)

  hh_start=STRMID(time_start, 12, 2)  ;;;to create time (hour) in the name of file of new image
  mm_start=STRMID(time_start, 15, 2)  ;;;to create time (min) in the name of file of new image
  ss_start=STRMID(time_start, 18, 2)  ;;;to create time (sec) in the name of file of new image



  ;;;create base difference images in a cycle
  for it=1,N-1 do begin
   
    print, it
    map_rot=drot_map(maps[it],REF_MAP=maps[0], /KEEP_LIMB)  ;;;differential rotation to base image,
 
    dmap=diff_map(map_rot,maps[0])  ;;;substract current image (that was differentially rotated) from base image

    time_cur=maps[it].time
    hh=STRMID(time_cur, 12, 2)   ;;;to create time (hour) in the name of file of new image
    mm=STRMID(time_cur, 15, 2)   ;;;to create time (min) in the name of file of new image
    ss=STRMID(time_cur, 18, 2)   ;;;to create time (sec) in the name of file of new image


    fname=''

    fname_save=fname+'BDif_'+hh+mm+ss+'_'+hh_start+mm_start+ss_start+'.sav'

    SAVE, dmap, FILENAME = path_save+fname_save
  endfor

ENDIF
step =4
IF step EQ 4 then BEGIN
  path_data=event + '/LASCO/base_diff_sav/'
  FILE_MKDIR, event + '/LASCO/base_diff_fits/'
  file_save=event + '/LASCO/base_diff_fits/'

  files = file_search(path_data +'*.sav' )

  N=n_elements(files)

  for it=0,N-1 do begin
    print,it
    restore,files[it]

    filename = STRMID(files[it],80 , 26)
    str_1 = STRSPLIT(files[it], '/\', /EXTRACT)  
    str_2 = str_1[-1].remove(-4)
   
    map=DMAP
    map.rtime=map.time
    map2fits,map,file_save+str_2+'.fits'
  endfor

endif

end