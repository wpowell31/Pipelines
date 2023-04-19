"""Pipe3 example."""
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
    "This is a course about Python list compression",
    candidate_labels = ["education", "politics", "business"]
)

print(res)
