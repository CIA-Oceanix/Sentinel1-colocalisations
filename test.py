import os
from utils.misc import log_print

for command in (
    "python colocalize.py    20170108t015819 --data NEXRAD_L2 --create_gif True           > test/01.NEXRAD.txt",
    "python colocalize.py    20200827t000933 --data ABI --channel C14 --create_gif True   > test/02.ABI_GOES.txt",
    "python colocalize.py    20210913t092920 --data ABI --channel C14 --create_gif True   > test/03.ABI_HIMAWARI.txt",
    "python colocalize.py    20200827t000933 --data RRQPEF --create_gif True              > test/04.RRQPEF_GOES.txt",
    "python colocalize.py    20210913t092920 --data RRQPEF --create_gif True              > test/05.RRQPEF_HIMAWARI.txt",
    "python colocalize.py    20200827t000933 --data GLM --create_gif True                 > test/06.GLM.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel DPR --create_gif True       > test/07.DPR.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel N0M --create_gif True       > test/08.N0M.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel N0H --create_gif True       > test/09.N0H.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel HHC --create_gif True       > test/10.HHC.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel N0Z --create_gif True       > test/11.N0Z.txt",
    "python colocalize_s3.py 20180117T180947                                              > test/12.S3.txt",
    
    "python colocalize.py    20200827t000933 --data ERA5 --channel northward_wind_at_10_metres --create_gif True --time_step 60 --max_timedelta 240 > test/13.ERA5-N.txt",
    "python colocalize.py    20200827t000933 --data ERA5 --channel eastward_wind_at_10_metres  --create_gif True --time_step 60 --max_timedelta 240 > test/14.ERA5-E.txt",

    "python process_s1.py 20170108t015819 NEXRAD             > test/15.NEXRAD.txt",
    "python process_s1.py 20180117t180947 BiologicalSlicks   > test/16.BiologicalSlicks.txt",
    ):
    log_print(command, 0, 1)
    os.system(command)
