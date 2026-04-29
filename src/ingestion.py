import pandas as pd
import logging

from config import RAW_DIR, FILES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    dataframes = {}

    for key, filename in FILES.items():
        file_path = RAW_DIR / filename
        if file_path.exists():
            dataframes[key] = pd.read_csv(file_path)
            logging.info(f'Loaded{key}: {dataframes[key].shape}')
        else:
            logging.error(f'Missing File: {filename} in {RAW_DIR}')
            raise FileNotFoundError(f'{filename} not found')
        
    return dataframes