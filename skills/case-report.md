---
name: case-report
description: Write a structured research report for one uploaded case.
---
Call predict, explain_prediction and find_similar_cases for the case, then write:

**Case <id> — model output**
- Predicted class and its probability; mention the runner-up class if it is above 10%.
- Top 3 drivers from explain_prediction, each in plain words (e.g. "lower intensity entropy than healthy tissue").
- Neighbour agreement: how many of the 5 nearest training images share the predicted label.
- Confidence note: flag low confidence if the top probability is below 70% or the neighbours disagree.

End with: "Research output only, not a diagnosis."
