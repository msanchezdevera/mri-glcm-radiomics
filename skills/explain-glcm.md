---
name: explain-glcm
description: Explain GLCM texture features to a non-technical person.
---
Explain without formulas:
- A GLCM counts how often pairs of neighbouring pixels have each pair of gray levels.
- Homogeneous tissue puts most counts near the diagonal (neighbours look alike): high homogeneity, low contrast.
- Heterogeneous tissue, which tumors often are, spreads the counts out: higher contrast and entropy.
- Use one concrete feature from the current case if a case is active (call explain_prediction).
Three short paragraphs at most.
