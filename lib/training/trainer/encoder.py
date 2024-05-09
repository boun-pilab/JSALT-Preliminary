import torch
import wandb

import torch.nn as nn
import torch.optim as optim

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from datasets import load_dataset

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1



class AutoTrainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001, epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        wandb.init(project="your_project_name", name="training_run")
        wandb.watch(self.model)


    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # compute masked indices
                batch_size, raw_sequence_length = input_values.shape
                sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
                mask_time_indices = _compute_mask_indices(
                    shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
                )
                sampled_negative_indices = _sample_negative_indices(
                    features_shape=(batch_size, sequence_length),
                    num_negatives=model.config.num_negatives,
                    mask_time_indices=mask_time_indices,
                )

                mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
                sampled_negative_indices = torch.tensor(
                    data=sampled_negative_indices, device=input_values.device, dtype=torch.long
                )

                outputs = self.model(
                    input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
                )

                cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)

            val_loss, val_acc = self.evaluate()

            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        wandb.finish()

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = correct / total

        return val_loss, val_acc