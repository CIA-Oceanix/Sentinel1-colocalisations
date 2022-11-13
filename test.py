import os
from utils import log_print

for command in (
    "python colocalize_nexrad.py           20170108t015819 --generate_gif True  > test/20170108t015819_nexrad.txt",
    "python colocalize_nexrad.py           20170108t015819 --generate_gif False > test/20170108t015819_nexrad_gifless.txt",
    "python colocalize_glm.py              20200827T000933  > test/20200827T000933_glm.txt",
    "python colocalize_goes_abi.py         20200827T000933 --generate_gif True  > test/20200827T000933_abi.txt",
    "python colocalize_goes_abi.py         20200827T000933 --generate_gif False > test/20200827T000933_abi_gifless.txt",
    "python colocalize_goes_rrqpe.py         20200827T000933 --generate_gif True  > test/20200827T000933_goes_rrqpe.txt",
    "python colocalize_goes_rrqpe.py         20200827T000933 --generate_gif False > test/20200827T000933_goes_rrqpe_gifless.txt",
    "python colocalize_himawari_rrqpe.py   20210913t092920 --generate_gif True  > test/20210913t092920_himawari_rrqpe.txt",
    "python colocalize_himawari_rrqpe.py   20210913t092920 --generate_gif False > test/20210913t092920_himawari_rrqpe_gifless.txt",
    "python colocalize_himawari_abi.py   20210913t092920 --generate_gif True  > test/20210913t092920_himawari_abi.txt",
    "python colocalize_himawari_abi.py   20210913t092920 --generate_gif False > test/20210913t092920_himawari_abi_gifless.txt"):
    log_print(command)
    os.system(command)
