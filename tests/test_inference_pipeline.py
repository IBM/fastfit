import unittest
from datasets import load_dataset
from fastfit import FastFitTrainer, FastFit, sample_dataset
from transformers import AutoTokenizer, pipeline

class TestInferencePipeline(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        self.dataset = load_dataset("FastFit/claim_stance_55")
        self.dataset["validation"] = self.dataset["test"]

        # Down sample the train data for 5-shot training
        self.dataset["train"] = sample_dataset(
            self.dataset["train"], label_column="label", num_samples_per_label=5
        )

        self.model_name = "sentence-transformers/paraphrase-mpnet-base-v2"

        model = FastFitTrainer(
            model_name_or_path=self.model_name,
            label_column_name="label",
            text_column_name="text",
            num_train_epochs=40,
            max_steps=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            max_text_length=32,
            dataloader_drop_last=False,
            num_repeats=4,
            optim="adafactor",
            clf_loss_factor=0.1,
            dataset=self.dataset,
        ).export_model()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def test_inference_pipeline_simple(self):
        result = self.classifier(self.dataset["validation"]["text"][0])[0]
        self.assertEqual(result["label"], 'year round schooling')

    def test_inference_pipeline_with_top_k(self):
        results = self.classifier(self.dataset["validation"]["text"][0], top_k=3)
        labels = [result["label"] for result in results]
        targets = ['year round schooling', 'physical education', 'raising the school leaving age to 18']
        self.assertEqual(labels, targets)

    def test_inference_pipeline_with_batch(self):
        print(self.classifier(self.dataset["validation"]["text"], batch_size=64))