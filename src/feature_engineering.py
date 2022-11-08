import pandas as pd
from config import Config
from logger import define_logger
import logging

# Create 2 new columns
def transaction_pct_rate(df:pd.DataFrame, logger:logging) -> pd.DataFrame:
    df['Amount_pct_to_sender'] = df.apply(lambda x: (x['amount']/x['oldbalanceOrg'])*100, axis=1)
    df['Amount_pct_to_beneficiary'] = df.apply(lambda x: (x['amount']/x['newbalanceDest'])*100, axis=1)
    return df
            
# Resolve rare occurrence categorical data
def resolve_rare_cat_val(df:pd.DataFrame, target:str, logger:logging) -> pd.DataFrame:
    for x in df.columns:
        rare_feature_merge = []
        # Object type columns only
        if df[x].dtypes == 'O':
            for y in df[x].value_counts().keys():
                
                # Check if occurrence <10 (rare data)
                if len(df[x].loc[(df[x]==y)]) < 10:
                    rare_feature_merge.append(y)

        # If distinct rare data > 1: group it as 'Other'
        if len(rare_feature_merge) >= 1:
            df[x] = df[x].apply(lambda x: 'Others' if x in rare_feature_merge else x)
            
            logger.info(str(len(rare_feature_merge)) + ' values in ' +str(x)+ ' are combined as [Others]')

        # Distinct rare data = 1
        # Remove the rare data if <10 occurrence and no contribute to the minority target
        if len(df[target].loc[(df[x] == 'Others')].value_counts()) >1: 
            y_count = len(df.loc[(df[x] == 'Others') & (df[target] == 1)])

            if y_count != 0:
                if y_count/(df.shape[0]) < 0.1:
                    df = df.loc[(df[x] != 'Others')]
            elif y_count == 0:
                df = df.loc[(df[x] != 'Others')]
    
    return df



if __name__ == '__main__':
    # Create directory
    Config.FEATURE_PATH.mkdir(parents=True, exist_ok= True)

    # Define the logger (no argument for continue write on latest log file)
    logger = define_logger()
    logger.info('##### Feature Engineering #####')

    # Load the 2 data from different directory
    train_df = pd.read_csv(Config.PROCESSED_DATASET_PATH / 'Processed_train.csv')
    test_df = pd.read_csv(Config.PROCESSED_DATASET_PATH / 'Processed_test.csv')

    # Store number of feautures before feature engineering
    ori_features = train_df.shape[1]

    # Run the functions for both dataset
    train_df = transaction_pct_rate(train_df, logger)
    train_df = resolve_rare_cat_val(train_df, 'isFraud', logger)
    test_df = transaction_pct_rate(test_df, logger)
    test_df = resolve_rare_cat_val(test_df, 'isFraud', logger)

    # Logging 
    logger.info('Generated '+ str(train_df.shape[1] - ori_features) +' new feature(s)')
    logger.info('New features: ' + str(train_df.columns[-2:]))
    


    # Extract features and label
    Train_features = train_df.drop(columns=['isFraud'])
    Train_label = train_df[['isFraud']]

    Test_features = test_df.drop(columns=['isFraud'])
    Test_label = test_df[['isFraud']]


    if Train_features.shape[0] == Train_label.shape[0] == train_df.shape[0] and Train_label.shape[1] == 1:
        if Test_features.shape[0] == Test_label.shape[0] == test_df.shape[0] and Test_label.shape[1] == 1:
            if Train_features.shape[1]  == Test_features.shape[1]:
                # Split and store into feature path
                Train_features.to_csv(Config.FEATURE_PATH /'Train_features.csv', index = False)
                Train_label.to_csv(Config.FEATURE_PATH / 'Train_label.csv', index = False)

                Test_features.to_csv(Config.FEATURE_PATH / 'Test_features.csv', index = False)
                Test_label.to_csv(Config.FEATURE_PATH / 'Test_label.csv', index = False)

                # Logging
                logger.info("Total features: "+ str(Train_features.shape[1]))
                logger.info('Dataset stored in ../../assets/features\n')

            else:
                logger.error('Data inconsistency\n')