import os
import unittest
from unittest.mock import patch
import numpy as np
from src.controllers.image_classification_controller import \
    ImageClassificationController
from src.views.image_classification_view import ImageClassificationView

class TestImageClassificationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Sets up the default test environment.
        """
        base_dir = os.path.dirname(__file__)
        cls.test_folder = os.path.join(base_dir, "test_env", "input_folder")
        cls.output_folder = os.path.join(base_dir, "test_env", "output_folder")
        if not os.path.exists(cls.test_folder):
            raise FileNotFoundError(
                f"Test folder not found: {cls.test_folder}")
        if not os.path.exists(cls.output_folder):
            os.makedirs(cls.output_folder)

    def setUp(self):
        """
        Initializes the test components.
        """
        self.view = ImageClassificationView()
        self.controller = ImageClassificationController(self.view)
        self.controller.output_directory = self.output_folder

    def test_batch_processing_workflow(self):
        """
        Tests the batch processing workflow.
        """
        expected_results = {
            "test_image_1.jpg": [
                ("miniature_poodle", 0.448),
                ("soft-coated_wheaten_terrier", 0.182),
                ("toy_poodle", 0.136)
            ],
            "test_image_3.jpg": [
                ("toucan", 0.753),
                ("macaw", 0.009),
                ("drake", 0.009)
            ],
            "test_image_2.jpg": [
                ("Egyptian_cat", 0.312),
                ("Siamese_cat", 0.227),
                ("lynx", 0.043)
            ],
        }
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

            def mock_classify(image, top_n):
                filename = os.path.basename(self.controller.current_image_path)
                if filename in expected_results:
                    return expected_results[filename]
                return []

            with patch.object(self.controller.model, 'classify_image', side_effect=mock_classify):
                self.controller.batch_classify_images(self.test_folder, 3)
                self.assertEqual(
                    len(self.controller.batch_results), len(expected_results))
                for result_text in self.controller.batch_results:
                    for filename, predictions in expected_results.items():
                        if filename in result_text:
                            for rank, (class_name, confidence) in enumerate(
                                    predictions, start=1):
                                expected_line = f"   {rank}. {class_name} ({confidence * 100:.1f}%)"
                                self.assertIn(expected_line, result_text)

    def test_single_image_classification(self):
        """
        Tests single image classification.
        """
        test_image = os.path.join(self.test_folder, "test_image_1.jpg")
        expected_predictions = [
            ("miniature_poodle", 0.448),
            ("soft-coated_wheaten_terrier", 0.182),
            ("toy_poodle", 0.136)
        ]
        with patch.object(self.controller.model, 'classify_image', return_value=expected_predictions):
            self.controller.classify_image(test_image, 3)
            for rank, (class_name, confidence) in enumerate(
                    expected_predictions, start=1):
                expected_text = f"{rank}. {class_name}\n   Confidence: {confidence * 100:.1f}%"
                self.assertIn(expected_text, self.view.result_label.text())

if __name__ == "__main__":
    unittest.main()