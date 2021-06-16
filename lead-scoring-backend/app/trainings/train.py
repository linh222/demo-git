import logging
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


root_path = get_project_root()

ver_1_0 = '1.0'


def load_training_dataset(model_version='1.0'):
    """
        This function is used to load the training dataset chosen with the following information:

        + Input:
            - file_path: the training file path.
        + Output:
            - features_data: The dataframe containing the features data for further steps.
            - label_data: the dataframe having the corresponding labels with respect to the features_data

        + The training file has the following attributes:
            - user_id
            - phone_type
            - TELCO
            - starting_09X
            - starting_08X
            - is_gmail
            - is_yahoo_email
            - is_educational_email
            - Package
            - Entity_Lead_Source
            - os_version
            - browser_type
        + Remarks: please filter all ELSA users from the training dataset as they are noise data that can affect the prediction results.

    """
    if model_version == ver_1_0:
        Package = ['lifetime_membership',
                   'one_year_credit',
                   'three_months_credit',
                   'six_months_credit',
                   'one_month_credit',
                   'N/A',
                   'group_membership',
                   'two_years_credit'
                   ]
        TELCO = [
            'MOBI',
            'VIETTEL',
            'VNM',
            'VINA',
            'N/A',
            'GM'
        ]
        phone_type = [
            'N/A',
            'Android',
            'iPhone',
            'PC',
            'MacBook'
        ]
        os_version = [
            'N/A',
            'Android 8',
            'Android 9',
            'OS 12_4',
            'Windows NT 10',
            'Android 10',
            'OS 14_2',
            'OS 14_3',
            'Windows NT 6.',
            'OS 13_3',
            'Android 5',
            'OS 13_7',
            'Android 7',
            'OS 14_0',
            'Android 1',
            'OS 11_4',
            'OS 13_5',
            'Android 6',
            'OS 10_3',
            'OS 13_6',
            'Android 4',
            'OS 12_5',
            'OS 13_1',
            'OS 14_1',
            'OS 11_2',
            'OS 11_0',
            'OS 12_3',
            'OS 14_4',
            'OS 13_2',
            'OS 12_1',
            'OS 12_0',
            'OS 12_2',
            'OS 9_2 ',
            'OS 13_4',
            'OS 11_3',
            'OS 9_2_',
            'OS 10_2',
            'Windows NT 5.',
            'OS 14_5',
            'OS 10_1',
            'OS 7_1']
        browser_type = [
            'N/A',
            'Chrome',
            'Safari',
            'Firefox',
            'Opera'
        ]
        Entity_Lead_Source = [
            'N/A',
            'Organic',
            'FB_Inbox',
            'FB_DynamicAds',
            'FB_FA4',
            'FB_FA1',
            'FB_FA2',
            'FB_FA3',
            'GG_YT',
            'GG_SEM',
            'FB_Chatbot',
            'AppSpecial',
            'FB_FA',
            'SaleTeam',
            'CCAds',
            'FB_FA5',
            'Affiliate',
            'FBLG_FA2',
            'FB_Chatbot_Paid',
            'FB_LG',
            'FB_Chatbot_Organic',
            'Promotions',
            'BD',
            'FB_Other',
            'FB_FA6',
            'UpSales',
            'Subscribes'
        ]
        file_path = "training_data/training_dataset_v1.csv"
        training_dataset = pd.read_csv(file_path, index_col=False)
        features_data = training_dataset[['phone_type',
                                          'TELCO',
                                          'starting_09X',
                                          'starting_08X',
                                          'is_gmail',
                                          'is_yahoo_email',
                                          'is_educational_email',
                                          'Package',
                                          'Entity_Lead_Source',
                                          'os_version',
                                          'browser_type']].copy()

        features_data = features_data.fillna('N/A')
        for i in features_data.select_dtypes(bool).columns:
            features_data.loc[:, i] = features_data.loc[:, i].apply(lambda x: 0 if x == False else 1)

        # Label encoding for the column Package
        label_encoder_package = preprocessing.LabelEncoder()
        label_encoder_package.fit(Package)
        features_data["Package"] = label_encoder_package.transform(features_data["Package"])

        # Label encoding for the column TELCO
        label_encoder_TELCO = preprocessing.LabelEncoder()
        label_encoder_TELCO.fit(TELCO)
        features_data["TELCO"] = label_encoder_TELCO.transform(features_data["TELCO"])

        # Label encoding for the column phone_type
        label_encoder_phone_type = preprocessing.LabelEncoder()
        label_encoder_phone_type.fit(phone_type)
        features_data["phone_type"] = label_encoder_phone_type.transform(features_data["phone_type"])

        # Label encoding for the column os_version
        label_encoder_os_version = preprocessing.LabelEncoder()
        label_encoder_os_version.fit(os_version)
        features_data["os_version"] = label_encoder_os_version.transform(features_data["os_version"])

        # Label encoding for the column browser_type
        label_encoder_browser_type = preprocessing.LabelEncoder()
        label_encoder_browser_type.fit(browser_type)
        features_data["browser_type"] = label_encoder_browser_type.transform(features_data["browser_type"])

        # Label encoding for the column Entity_Lead_Source
        label_encoder_Entity_Lead_Source = preprocessing.LabelEncoder()
        label_encoder_Entity_Lead_Source.fit(Entity_Lead_Source)
        features_data["Entity_Lead_Source"] = label_encoder_Entity_Lead_Source.transform(
            features_data["Entity_Lead_Source"])

        label_data = training_dataset.loc[:, "Is_Converted"]
        label_vector = [1 if i == 'Yes' else 0 for i in list(label_data)]
        return features_data, label_vector
    else:
        logging.info("The latest base_model has not supported for this version!")
        features_data = []
        label_vector = []
        return features_data, label_vector


def feature_validation(feature_vector, model_version='1.0'):
    """
        This function is used to validate the values of a computed feature vector depending on the base_model version.
        Here, the feature vector has the following fields:
        - phone_type
        - TELCO
        - starting_09X
        - starting_08X
        - is_gmail
        - is_yahoo_email
        - is_educational_email
        - Package
        - Entity_Lead_Source
        - os_version
        - browser_type
        For instance:
            feature_vector = [
                'Android,
                'VIETTEL',
                True,
                False,
                True,
                False,
                False,
                'one_year_credit',
                'Organic',
                'Android 9',
                'Safari'
            ]

    """

    if model_version == ver_1_0:

        # The default values of chosen features (not Bool variables) in the base_model version 1.0

        Package = ['lifetime_membership',
                   'one_year_credit',
                   'three_months_credit',
                   'six_months_credit',
                   'one_month_credit',
                   'N/A',
                   'group_membership',
                   'two_years_credit'
                   ]
        TELCO = [
            'MOBI',
            'VIETTEL',
            'VNM',
            'VINA',
            'N/A',
            'GM'
        ]
        phone_type = [
            'N/A',
            'Android',
            'iPhone',
            'PC',
            'MacBook'
        ]
        os_version = [
            'N/A',
            'Android 8',
            'Android 9',
            'OS 12_4',
            'Windows NT 10',
            'Android 10',
            'OS 14_2',
            'OS 14_3',
            'Windows NT 6.',
            'OS 13_3',
            'Android 5',
            'OS 13_7',
            'Android 7',
            'OS 14_0',
            'Android 1',
            'OS 11_4',
            'OS 13_5',
            'Android 6',
            'OS 10_3',
            'OS 13_6',
            'Android 4',
            'OS 12_5',
            'OS 13_1',
            'OS 14_1',
            'OS 11_2',
            'OS 11_0',
            'OS 12_3',
            'OS 14_4',
            'OS 13_2',
            'OS 12_1',
            'OS 12_0',
            'OS 12_2',
            'OS 9_2 ',
            'OS 13_4',
            'OS 11_3',
            'OS 9_2_',
            'OS 10_2',
            'Windows NT 5.',
            'OS 14_5',
            'OS 10_1',
            'OS 7_1']
        browser_type = [
            'N/A',
            'Chrome',
            'Safari',
            'Firefox',
            'Opera'
        ]
        Entity_Lead_Source = [
            'N/A',
            'Organic',
            'FB_Inbox',
            'FB_DynamicAds',
            'FB_FA4',
            'FB_FA1',
            'FB_FA2',
            'FB_FA3',
            'GG_YT',
            'GG_SEM',
            'FB_Chatbot',
            'AppSpecial',
            'FB_FA',
            'SaleTeam',
            'CCAds',
            'FB_FA5',
            'Affiliate',
            'FBLG_FA2',
            'FB_Chatbot_Paid',
            'FB_LG',
            'FB_Chatbot_Organic',
            'Promotions',
            'BD',
            'FB_Other',
            'FB_FA6',
            'UpSales',
            'Subscribes'
        ]

        # Label encoding for the column Package
        label_encoder_package = preprocessing.LabelEncoder()
        label_encoder_package.fit(Package)

        # Label encoding for the column TELCO
        label_encoder_TELCO = preprocessing.LabelEncoder()
        label_encoder_TELCO.fit(TELCO)

        # Label encoding for the column phone_type
        label_encoder_phone_type = preprocessing.LabelEncoder()
        label_encoder_phone_type.fit(phone_type)

        # Label encoding for the column os_version
        label_encoder_os_version = preprocessing.LabelEncoder()
        label_encoder_os_version.fit(os_version)

        # Label encoding for the column browser_type
        label_encoder_browser_type = preprocessing.LabelEncoder()
        label_encoder_browser_type.fit(browser_type)

        # Label encoding for the column Entity_Lead_Source
        label_encoder_Entity_Lead_Source = preprocessing.LabelEncoder()
        label_encoder_Entity_Lead_Source.fit(Entity_Lead_Source)

        # Validate the feature vector:
        validated_feature_vector = feature_vector

        if validated_feature_vector[0] not in phone_type:  # phone_type
            validated_feature_vector[0] = 'N/A'

        if validated_feature_vector[1] not in TELCO:  # TELCO
            validated_feature_vector[1] = 'N/A'

        if validated_feature_vector[7] not in Package:  # Package
            validated_feature_vector[7] = 'N/A'

        if validated_feature_vector[8] not in Entity_Lead_Source:  # Entity_Lead_Source
            validated_feature_vector[8] = 'N/A'

        if validated_feature_vector[9] not in os_version:  # os_version
            validated_feature_vector[9] = 'N/A'

        if validated_feature_vector[10] not in browser_type:  # browser_type
            validated_feature_vector[10] = 'N/A'

        validated_feature_vector = list(validated_feature_vector)
        # Encode the feature vector with the chosen base_model version

        validated_feature_vector[0] = label_encoder_phone_type.transform([validated_feature_vector[0]])[0]
        validated_feature_vector[1] = label_encoder_TELCO.transform([validated_feature_vector[1]])[0]
        for i in range(2, 7):
            if validated_feature_vector[i] == 'True' or validated_feature_vector[i] == True:
                validated_feature_vector[i] = 1
            else:
                validated_feature_vector[i] = 0
        validated_feature_vector[7] = label_encoder_package.transform([validated_feature_vector[7]])[0]
        validated_feature_vector[8] = label_encoder_Entity_Lead_Source.transform([validated_feature_vector[8]])[0]
        validated_feature_vector[9] = label_encoder_os_version.transform([validated_feature_vector[9]])[0]
        validated_feature_vector[10] = label_encoder_browser_type.transform([validated_feature_vector[10]])[0]

        return validated_feature_vector

    else:
        logging.info("The latest base_model has not supported for this version!")
        validated_feature_vector = []
        return validated_feature_vector


def model_training(model_version='1.0'):
    """
        This function is used to train the corresponding base_model based on the chosen version
         and store the trained one into a Pickle file.
    """

    if model_version == ver_1_0:
        features_data, label_vector = load_training_dataset(model_version)
        if len(features_data):
            X_train = features_data
            y_train = label_vector

            clf = RandomForestClassifier(max_depth=50).fit(features_data, label_vector)

            predict_train = clf.predict(X_train)
            logging.info("Train")
            logging.info("accuracy: ", accuracy_score(np.array(predict_train), y_train))
            logging.info("f1 score", f1_score(np.array(predict_train), y_train))
            logging.info("recall score", recall_score(np.array(predict_train), y_train))
            logging.info("precision score", precision_score(np.array(predict_train), y_train))

            # Save to file in the current working directory
            model_filename = "{}/models/lead_scoring_model_version_{}.pkl".format(root_path, model_version)
            with open(model_filename, 'wb') as file:
                pickle.dump(clf, file)
                logging.info("Complete saving the version {} of the lead scoring base_model at the following file: {}".format(
                    model_version, model_filename))
                logging.info("--------------------------------------")

    else:
        logging.info("The latest base_model has not supported for this version!")


def validate_feature(feature_vectors, model_version='1.0'):
    """
        This function is used to validate list feature vectors with the chosen version:
        - Input data:
            feature_vectors =
    """
    if model_version == '1.0':

        # Prepare the list of predicted feature vectors after the valdiation step:
        logging.info("--------------------------------------")
        logging.info("Doing the validation steps...")
        predicted_feature_vectors = []
        for feature_vector in feature_vectors:
            validated_feature_vector = feature_validation(feature_vector, model_version)
            logging.info("Before: {}".format(feature_vector))
            logging.info("After: {}".format(validated_feature_vector))
            predicted_feature_vectors.append(validated_feature_vector)
            logging.info("********************")
        return predicted_feature_vectors
    else:
        logging.info("Not supported for this version!")
        return []


def predict(predicted_feature_vectors, model_version='1.0'):
    """
        This function is used to predict an arraye of feature vectors using the trained base_model
        with the chosen version:
        - Input data:
            predicted_feature_vectors =
    """

    if model_version == ver_1_0:

        # Load the corresponding base_model trained from file
        model_filename = "{}/models/lead_scoring_model_version_{}.pkl".format(root_path, model_version)
        with open(model_filename, 'rb') as file:
            trained_model = pickle.load(file)
            logging.info("Complete loading the version {} of the lead scoring base_model from the following file: {}".format(
                model_version, model_filename))
            logging.info("--------------------------------------")

        logging.info("Starting doing the prediction for all input feature vectors...")
        Results = trained_model.predict_proba(predicted_feature_vectors)

        predicted_probability = []
        for result in Results:
            # Return the predicted probability of label = 1 (which means "Converted")
            predicted_probability.append(result[1])
        logging.info("Completed the prediction for all input feature vectors...")
        logging.info("--------------------------------------")
        return predicted_probability

    else:
        logging.info("The latest base_model has not supported for this version!")
        return []


if __name__ == "__main__":

    # Choose the base_model version:
    model_version = ver_1_0

    # Training the chosen version of the lead scoring base_model:
    model_training(model_version)

    # Testing the trained base_model:
    feature_vectors = [
        [
            'Android',
            'VIETTEL',
            'True',
            'False',
            'True',
            'False',
            'False',
            'one_year_credit',
            'Organic',
            'Android 9',
            'Safari'
        ],
        [
            'iPhone',
            'Mobi',
            'True',
            'False',
            'True',
            'False',
            'False',
            'one_year_credit',
            'N/A',
            'N/A',
            'N/A'
        ]
    ]

    feature_validated = validate_feature(feature_vectors, model_version='1.0')
    predicted_probability = predict(feature_validated, model_version='1.0')
    logging.info("--------------------------------------")
    logging.info("The predicted probability of the testing feature vectors as:")
    for i in range(0, len(feature_vectors)):
        feature_vector = feature_vectors[i]
        logging.info("The feature vector: {}".format(feature_vector))
        logging.info("has the following predicted probability: {}".format(predicted_probability[i]))
        logging.info("******************")
