import logging 
from config import Config
import datetime
import os

def define_logger(create_new:bool = False):
    # Now we will create and configure logger 
    f_name = latest_log(create_new)
    logging.basicConfig(filename = Config.LOG_PATH / str(datetime.datetime.now().strftime("%d_%m_%Y")) / f_name, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='a') 

    # Let us Create an object 
    logger=logging.getLogger() 

    # Now we are going to Set the threshold of logger to DEBUG 
    logger.setLevel(logging.DEBUG) 

    return logger


def latest_log(create_new:bool = False, pattern: str = "*"):
    # Create directory
    Config.LOG_PATH.mkdir(parents= True, exist_ok=True)

    # Create sub directory for today log folder if not exists (today date)
    sub_dir = os.path.join(Config.LOG_PATH, str(datetime.datetime.now().strftime("%d_%m_%Y")))

    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    path = Config.LOG_PATH / str(datetime.datetime.now().strftime("%d_%m_%Y"))
    files = path.glob(pattern)

    try:
        f_path = str(max(files, key=lambda x: x.stat().st_ctime))
        f_name = f_path[f_path.rindex('\\')+1:]
        num = [int(x) for x in f_name if x.isdigit()] 
        if create_new == True:
            # Create a +1 version on the latest found file
            return 'v' + str((sum(d * 10**i for i, d in enumerate(num[::-1])))+1) +'.log'
            
        elif create_new == False:
            # Use the found latest file
            return 'v' + str((sum(d * 10**i for i, d in enumerate(num[::-1])))) +'.log'

    except ValueError:
        return 'v1.log'

