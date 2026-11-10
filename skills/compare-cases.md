---
name: compare-cases
description: Compare two uploaded cases side by side.
---
Call predict and explain_prediction for both cases. Then answer in this order:
1. One line per case: predicted class and probability.
2. Whether the model separates them clearly (different classes, or same class with probabilities more than 20 points apart).
3. The two or three features where the cases differ most, using the drivers lists.
Keep it under 120 words.
