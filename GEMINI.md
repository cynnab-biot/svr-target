## Remaining Task: "100 best" selection for flexibility calculation

The current implementation selects the "top 100" compounds for which flexibility is calculated based on their `Predicted pIC50` values (highest predicted potency).

The request states: "100 best are categorized by being most similar according to morgan scores - these are specific target relative reference drivers."

Please clarify how these "100 best" compounds should be selected:

1.  **Based on a Reference Compound:** Should the "100 best" be the 100 compounds with the highest Morgan similarity to a specific reference compound? If so, how is this reference compound determined (e.g., the most potent compound, a user-defined compound)?
2.  **Diverse Selection within Clusters:** Should the "100 best" be a diverse selection of compounds (e.g., representative compounds from each cluster) that also exhibit high potency or similarity?
3.  **Other Criteria:** Is there another method for selecting the "100 best" based on Morgan similarity that should be implemented?
