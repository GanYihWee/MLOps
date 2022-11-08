import pandas as pd
from config import Config
from logger import define_logger
import logging


# Remove duplicates
def resolve_duplicate(df:pd.DataFrame, logger:logging) -> pd.DataFrame:

    Before = df[df.duplicated()].shape[0]
    df.drop_duplicates(inplace=True)
    After = df[df.duplicated()].shape[0]

    # Logging
    if Before == 0:
        logger.info('No duplicates.')
    elif Before != 0 and After == 0:
        logger.info('Dropped ' + str(Before) +' duplicates.')
    elif Before!=0 and After != 0:
        logger.warning('Left ' + str(After) +' duplicates.')
    elif Before == After:
        logger.warning(str(After) +' duplicates remain.')

    return df

# Resolve missing values
def missing_values(df: pd.DataFrame, target:str, logger:logging) -> pd.DataFrame:

    if sum(df.isnull().sum().tolist()) == 0:
        logger.info('No missing values.')
    else:
        for x in df.columns:
            # If null exits and <1%:
            if df[x].isnull().sum() !=0 and df[x].isnull().sum()/df.shape[0] < 0.1:
                # if null exits contribute to <1% of minority target
                if len(df.loc[(df[x].isnull() == True) & df[target] == 1])/len(df.loc[df[target] == 1]) <0.01:
                    df.dropna(subset=[x])

        # Logging
        if sum(df.isnull().sum().tolist()) == 0:
            logger.info('All missing values are resolved.')
        else:
            logger.warning('Found ' + str(sum(df.isnull().sum().tolist()) + ' missing values.'))

    return df

# Data consistency check
def data_consistency_impute(df:pd.DataFrame, logger:logging) -> pd.DataFrame:
    ori_size = df.shape[0]

    # Only rows with balance and valid transaction
    # Remove rows with zero amount for transaction
    # Remove when he/she has 0 balance
    df = df.loc[(df['amount'] != 0) | (df['oldbalanceOrg'] !=0)]

    # Amount is less or equal to their current balance
    df = df.loc[(df['amount'] <= df['oldbalanceOrg'])]

    # New balance of sender will be deducted
    # New balance for beneficial will be increased
    df['newbalanceOrg'] = df.apply(lambda x: x['oldbalanceOrg']-x['amount'], axis=1)
    df['newbalanceDest'] = df.apply(lambda x: x['oldbalanceDest']+x['amount'], axis=1)

    processed_size = df.shape[0]

    # Logging
    logger.warning('Removed '+ str(ori_size-processed_size) +' rows ('+str((processed_size/ori_size)*100)+'%).')
    logger.info('New dataframe shape '+ str(df.shape))

    return df



if __name__ == '__main__':

    # Create directory
    Config.PROCESSED_DATASET_PATH.mkdir(parents=True, exist_ok= True)
    
    # Define the logger (no arguement for continue write on latest log file)
    logger = define_logger()

    logger.info('##### Preprocessing #####')

    # Load the train test splitted data
    df_train = pd.read_csv(Config.TRAIN_TEST_PATH / 'Train.csv')
    df_test = pd.read_csv(Config.TRAIN_TEST_PATH / 'Test.csv')

    # Run all the functions above for train set (preprocessing)
    logger.info('[Train]')

    # logger.info('Droping Duplicates ... ')
    # df_train = resolve_duplicate(df_train)
    logger.info('Resolving Missing value(s) ... ')
    df_train = missing_values(df_train, 'isFraud', logger)
    logger.info('Data consistency imputation ... ')
    df_train = data_consistency_impute(df_train, logger)

    # Run drop duplicates and data consistency check only for test set
    logger.info('[Test]')
    logger.info('Data consistency imputation ... ')
    df_test = data_consistency_impute(df_test, logger)

    logger.info("Processed Train shape: "+ str(df_train.shape))
    logger.info("Processed Test shape: "+ str(df_test.shape))


    # Store into desired directory
    if df_train.shape[1] == df_test.shape[1] and df_train.shape[0] > df_test.shape[0]:

        df_train.to_csv(Config.PROCESSED_DATASET_PATH / 'Processed_Train.csv', index=False)
        df_test.to_csv(Config.PROCESSED_DATASET_PATH / 'Processed_Test.csv', index=False)

        # Logging
        logger.info('Processed dataset stored in ../../assets/processed\n')

    else:
        logger.error('Data inconsistency\n' +'\n')