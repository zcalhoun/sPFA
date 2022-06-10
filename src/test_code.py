import os
from utils import DataHandler
import logging


class TestDataHandler:
    def test_by_file(self):

        logging.basicConfig(filename="./example.log", level=logging.DEBUG)
        logging.info(os.getcwd())
        d = DataHandler(
            "../../data/lemmatized",
            "../../data/storage",
            "by_file",
            100,
            10,
            100,
            0.01,
            "portland",
            "portland",
        )

        # Get the training data
        train_data = d.get_train_data()

        # Confirm that the training data
        # contains 30 documents
        assert len(train_data) == 30

        # Confirm that the files are in the
        # destination folder
        assert os.path.exists("../../data/storage/train_data.joblib")

        test_data = d.get_test_data()
        assert len(test_data) == 30

        assert os.path.exists("../../data/storage/test_data.joblib")
        # Delete the created files
        os.remove("../../data/storage/train_data.joblib")
        os.remove("../../data/storage/test_data.joblib")

    def test_by_day(self):
        d = DataHandler(
            "../../data/lemmatized",
            "../../data/storage",
            "by_day",
            1000,
            10,
            100,
            0.01,
            "portland",
            "portland",
        )

        # Get the training data
        train_data = d.get_train_data()
        test_data = d.get_test_data()
        # Confirm that the training data
        # contains 104 documents in this case.
        assert len(train_data) == 104
        assert len(test_data) == 104
        # Confirm that the files are in the
        # destination folder
        assert os.path.exists("../../data/storage/train_data.joblib")
        assert os.path.exists("../../data/storage/test_data.joblib")
        # Delete the created files
        os.remove("../../data/storage/train_data.joblib")
        os.remove("../../data/storage/test_data.joblib")
