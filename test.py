import os
from utils.misc import log_print

for command in (
    "python colocalize_abi.py   20170108t015819 --platform_key nexrad --generate_gif True  > test/20170108t015819_nexrad.txt",
    "python colocalize_abi.py   20170108t015819 --platform_key nexrad --generate_gif False > test/20170108t015819_nexrad_gifless.txt",
    "python colocalize_abi.py   20200827T000933 --platform_key abi --channel C14 --generate_gif True  > test/20200827T000933_abi.txt",
    "python colocalize_abi.py   20200827T000933 --platform_key abi --channel C14 --generate_gif False > test/20200827T000933_abi_gifless.txt",
    "python colocalize_abi.py   20210913t092920 --platform_key abi --channel C14 --generate_gif True  > test/20210913t092920_himawari_abi.txt",
    "python colocalize_abi.py   20210913t092920 --platform_key abi --channel C14 --generate_gif False > test/20210913t092920_himawari_abi_gifless.txt"):
    log_print(command)
    os.system(command)
