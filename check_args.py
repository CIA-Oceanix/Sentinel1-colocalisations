
GOES_SERIE = ["goes16", "goes17", "goes18"]
HIMAWARI_SERIE = ["himawari8", "himawari9"]
NEXRAD_BASIS = ["nexrad-level2"]
SATELLITE_PLATFORMS = GOES_SERIE + HIMAWARI_SERIE

ABI_CHANNELS = ['C13', 'C14'] # C01 to C16 are available, but would need other cmap
RRQPEF_CHANNELS = ['RRQPEF']
GLM_CHANNELS = ['GLM']

with open('res/nexrad_stations.txt', 'r') as file:
    NEXRAD_CHANNELS = [line.split('\t')[1] for line in file.readlines()[1:]]

def check_args(key=None, channel=None, shape=None, metadata_filename=None, verbose=None, sensoroperationalmode=None, platform_key=None, gif=True, max_timedelta=None, time_step=None):
        
    # Set default values
    if key is None: key = "20170108t015819"
    if channel is None: channel = None#"C14"
    if verbose is None: verbose = 1
    if sensoroperationalmode is None: sensoroperationalmode = 'IW'
    if platform_key is None: platform_key = "nexrad"
    if gif is None: gif = True
    if max_timedelta is None: max_timedelta= 90
    if time_step is None: time_step= 10

    
    if key.endswith('.txt'):
        assert os.path.exists(key)
        with open(key, 'r') as file:
            lines = file.readlines()
            key = [line.replace('\n', 'r') for line in lines]
    else:
        keys = [key]

    if platform_key == 'nexrad':
        platforms = NEXRAD_BASIS
    elif platform_key in ('abi', "rrqpef"):
        platforms = SATELLITE_PLATFORMS
    elif platform_key in ('goes', 'glm'):
        platforms = GOES_SERIE
    elif platform_key == 'himawari':
        platforms = HIMAWARI_SERIE
    else:
        raise ValueError

    #if platform_key == "nexrad": channel = None
    if platform_key == "glm": channel = "GLM"
    if platform_key == "rrqpef": channel = "RRQPEF"

    return keys, channel, shape, metadata_filename, verbose, sensoroperationalmode, platforms, gif, max_timedelta, time_step
    
