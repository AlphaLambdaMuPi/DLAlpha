import logging, logging.handlers, os, sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))                           
LOG_FILE_NAME = os.path.join(ROOT_DIR, 'log/server.log')          
LOG_LEVEL = logging.DEBUG 
def init_log_settings():                                                        

    log_parent = os.path.dirname(LOG_FILE_NAME)                                 
    if not os.path.isdir(log_parent):                                           
        os.makedirs(log_parent)                                                 

    rotator = logging.handlers.RotatingFileHandler(                             
        LOG_FILE_NAME, maxBytes=65536, backupCount=5)                 
    logging.basicConfig(handlers=[rotator],                                     
        format='[%(asctime)s.%(msecs)d] %(module)s - %(levelname)s : %(message)s',
        datefmt='%H:%M:%S',                                     
        level=LOG_LEVEL)                                        
    logger = logging.getLogger()                                                

    formatter = logging.Formatter(fmt = '[%(asctime)s.%(msecs)d] %(module)s - %(levelname)s : %(message)s'
                ,datefmt = '%H:%M:%S')                        
    console = logging.StreamHandler(sys.stdout)                                 
    console.setLevel(logging.DEBUG)                                             
    console.setFormatter(formatter)                                             
    logger.addHandler(console)
