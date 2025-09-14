import os, logging

def setup_logger(log_file='run_example.log'):
    logs_path = r'./logs'
    os.makedirs(logs_path, exist_ok=True)
    log_path = os.path.join(logs_path, log_file)
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s [%(levelname)s]  %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO
    )
    return logging.getLogger()