import unittest
from datasets import load_dataset
from fastfit import FastFitTrainer, sample_dataset

class TestFullTrain(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        self.dataset = load_dataset("FastFit/claim_stance_55")
        self.dataset["validation"] = self.dataset["test"]

        # Down sample the train data for 5-shot training
        self.dataset["train"] = sample_dataset(
            self.dataset["train"], label_column="label", num_samples_per_label=5
        )

        # Setup the trainer
        self.trainer = FastFitTrainer(
            model_name_or_path="sentence-transformers/paraphrase-mpnet-base-v2",
            label_column_name="label",
            text_column_name="text",
            num_train_epochs=40,
            max_steps=10,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            max_text_length=32,
            dataloader_drop_last=False,
            num_repeats=4,
            optim="adafactor",
            clf_loss_factor=0.1,
            dataset=self.dataset,
        )

    def test_full_train(self):
        # Train the model
        model = self.trainer.train()

        # Evaluate the model
        results = self.trainer.evaluate()

        # Check if eval_accuracy is between 82% and 84%
        accuracy = results['eval_accuracy']
        self.assertTrue(0.82 <= accuracy <= 0.84,
                        f"The evaluation accuracy {accuracy*100:.2f}% is not within the expected range of 82% to 84%.")

if __name__ == "__main__":
    unittest.main()
