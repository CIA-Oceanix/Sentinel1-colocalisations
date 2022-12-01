import os
from utils.misc import log_print

for command in (
    "python colocalize.py    20170108t015819 --platform_key nexrad --create_gif True             > test/20170108t015819_nexrad.txt",
    "python colocalize.py    20200827t000933 --platform_key abi --channel C14 --create_gif True  > test/20170108t015819_abi.txt",
    "python colocalize.py    20210913t092920 --platform_key abi --channel C14 --create_gif True  > test/20200827T000933_abi.txt",
    "python colocalize.py    20200827t000933 --platform_key rrqpef --create_gif True             > test/20200827T000933_rrqpef.txt",
    "python colocalize.py    20210913t092920 --platform_key rrqpef --create_gif True             > test/20210913t092920_rrqpef.txt",
    "python colocalize.py    20200827t000933 --platform_key glm --create_gif True                > test/20200827t000933_glm.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel DPR --create_gif True              > test/20170108t015819_DPR.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel N0M --create_gif True              > test/20170108t015819_N0M.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel N0H --create_gif True              > test/20170108t015819_N0H.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel HHC --create_gif True              > test/20170108t015819_HHC.txt",
    "python colocalize_nexrad_l3.py 20170108t015819 --channel N0Z --create_gif True              > test/20170108t015819_N0Z.txt",
    "python colocalize_s3.py 20180117T180947 > test/20180117T180947_s3.txt                       > test/20180117T180947_s3.txt",
    
    "python colocalize.py    20200827t000933 --platform_key era5 --channel northward_wind_at_10_metres --create_gif True                > test/20200827t000933_era5-n.txt",
    "python colocalize.py    20200827t000933 --platform_key era5 --channel eastward_wind_at_10_metres  --create_gif True                > test/20200827t000933_era5-e.txt",
    ):
    log_print(command)
    os.system(command)
