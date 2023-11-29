'''
This script is used for testing and logging customer churn prediction.

Author: Salma Borchani

Date: 5th September 2023
'''

# Import necessary libraries
import os
import logging
import churn_library as cl

# Configure basic logging settings
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


def test_import():
    '''
    Test the import_data() function from the churn_library module
    '''
    # Test if the CSV file is available
    try:
        dataframe = cl.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file was not found")
        raise err

    # Test the structure of the dataframe
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info(
            'Rows: %d\tColumns: %d',
            dataframe.shape[0],
            dataframe.shape[1])
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file does not appear to have rows and columns")
        raise err


def test_eda():
    '''
    Test the perform_eda() function
    '''
    bank_df = cl.import_data(pth='./data/bank_data.csv')

    try:
        # Perform Exploratory Data Analysis (EDA)
        eda_df = cl.perform_eda(dataframe=bank_df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda failed - Error type: %s", type(err))

    image_files = [
        "churn_distribution.png",
        "customer_age_distribution.png",
        "marital_status_distribution.png",
        "total_trans_ct_distribution.png",
        "heatmap.png"
    ]

    for image_file in image_files:
        file_path = f"./images/eda/{image_file}"
        assert os.path.isfile(file_path) is True


def test_encoder_helper():
    '''
    Test the encoder_helper() function from the churn_library module
    '''
    # Load DataFrame
    dataframe = cl.import_data("./data/bank_data.csv")

    # Create `Churn` feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Categorical Features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    try:
        encoded_df = cl.encoder_helper(
            dataframe=dataframe, category_lst=[], response=None)

        # Data should remain the same
        assert encoded_df.equals(dataframe) is True
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=[]): SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=[]): ERROR")
        raise err

    try:
        encoded_df = cl.encoder_helper(
            dataframe=dataframe,
            category_lst=cat_columns,
            response=None)

        # Column names should remain the same
        assert encoded_df.columns.equals(dataframe.columns) is True

        # Data should be different
        assert encoded_df.equals(dataframe) is False
        logging.info(
            "Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: ERROR")
        raise err

    try:
        encoded_df = cl.encoder_helper(
            dataframe=dataframe,
            category_lst=cat_columns,
            response='Churn')

        # Columns names should be different
        assert encoded_df.columns.equals(dataframe.columns) is False

        # Data should be different
        assert encoded_df.equals(dataframe) is False

        # Number of columns in encoded_df should be the sum of columns in
        # dataframe and the newly created columns from cat_columns
        assert len(
            encoded_df.columns) == len(
            dataframe.columns) + len(cat_columns)
        logging.info(
            "Testing encoder_helper with response='Churn': SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper with response='Churn': ERROR")
        raise err


def test_perform_feature_engineering():
    '''
    Test perform_feature_engineering()
    '''
    two_test_level = False

    try:
        dataframe = cl.import_data(pth='./data/bank_data.csv')
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            dataframe)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering failed - Error type: %s",
            type(err))

    if two_test_level:
        try:
            assert x_train.shape[0] > 0
            assert x_train.shape[1] > 0
            assert x_test.shape[0] > 0
            assert x_test.shape[1] > 0
            assert y_train.shape[0] > 0
            assert y_test.shape[0] > 0
            logging.info(
                "perform_feature_engineering returned Train and Test sets of shape %s %s",
                x_train.shape,
                x_test.shape)
        except AssertionError:
            logging.error(
                "The returned train datasets do not have rows and columns")


def test_train_models():
    '''
    Test train_models() function from the churn_library module
    '''
    # Load the DataFrame
    dataframe = cl.import_data("./data/bank_data.csv")

    # Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Feature engineering
    (x_train, x_test, y_train, y_test) = cl.perform_feature_engineering(
        dataframe=dataframe, response='Churn')

    # Assert if `logistic_model.pkl` file is present
    try:
        cl.train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
    except AssertionError as err:
        logging.error("No such file on disk")
        raise err

    # Assert if `rfc_model.pkl` file is present
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
    except AssertionError as err:
        logging.error("No such file on disk")
        raise err

    # Assert if `roc_curve_result.png` file is present
    try:
        assert os.path.isfile('./images/results/roc_curve_result.png') is True
    except AssertionError as err:
        logging.error("No such file on disk")
        raise err

    # Assert if `rfc_results.png` file is present
    try:
        assert os.path.isfile('./images/results/rf_results.png') is True
    except AssertionError as err:
        logging.error("No such file on disk")
        raise err

    # Assert if `logistic_results.png` file is present
    try:
        assert os.path.isfile('./images/results/logistic_results.png') is True
    except AssertionError as err:
        logging.error("No such file on disk")
        raise err

    # Assert if `feature_importances.png` file is present
    try:
        assert os.path.isfile(
            './images/results/feature_importances.png') is True
    except AssertionError as err:
        logging.error("No such file on disk")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()