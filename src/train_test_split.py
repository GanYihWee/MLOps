from sklearn.model_selection import train_test_split
import pandas as pd
from config import Config
from logger import define_logger
import gdown
from preprocessing import resolve_duplicate

# Define the logger. True for create a new log file
logger = define_logger(True)

logger.info('###### Train Test Split ######')
# Create directory
Config.ORIGINAL_DATASET_PATH.parent.mkdir(parents =True, exist_ok= True) #parent = True: create parent dir if not exist
Config.TRAIN_TEST_PATH.mkdir(parents = True, exist_ok= True)

# Download dataset
gdown.download(
    "https://drive.google.com/uc?export=download&id=1XFHpCrzi3VIavxfHWt5Wj9b3EC-1sGh9",
    str(Config.ORIGINAL_DATASET_PATH)
)

# Load dataset
try:
    df = pd.read_csv(str(Config.ORIGINAL_DATASET_PATH))
except Exception as e:
    logger.exception(
        "Unable to load CSV, check your directory. Error: %s", e)
        
# Logging
if df.shape[0] == 0:
    logger.warning("The dataset has 0 row.")
elif df.shape[1] == 0:
    logger.warning("The dataset has 0 column.")
else:
    logger.info("Data is loaded succesfully.")

df = df.drop(columns=['nameOrig','nameDest'])

# Drop duplicates
logger.info('Droping Duplicates ... ')
df = resolve_duplicate(df ,logger)

# Train test split
Train_df, Test_df = train_test_split(df, shuffle= True, test_size = 0.2, 
                                     stratify= df['isFraud'], random_state= Config.RANDOM_SEED)



if Train_df.shape[0] != 0 and Test_df.shape[0]!=0 and Train_df.shape[0] > Test_df.shape[0] and Train_df.shape[0] + Test_df.shape[0] == df.shape[0]:
    # Store to desired directory
    Train_df.to_csv(Config.TRAIN_TEST_PATH / 'Train.csv', index=False)
    Test_df.to_csv(Config.TRAIN_TEST_PATH / 'Test.csv', index=False) 

    # Logging
    logger.info("Train shape: "+ str(Train_df.shape))
    logger.info("Test shape: "+ str(Test_df.shape))
    logger.info("Train test splitted sucessfullt at ../../assets/train_test_split" +'\n')

else:
    logger.error('Data quantity inconsistency\n')



