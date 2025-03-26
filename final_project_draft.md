

# Abstract
Rapid responce 
Easily tunable and explainable
We made toy example


# Introduction
Many methods of LLM guard (already in early report)
We chose external classifier.
- For it's model agnostic

Need lot's of nuanced data - let's generate it!

# Literature review
Retell the anthropic paper:
- Take constitutions
- Generate good/bad data with unguarded LLM
- Train classifier

# Experiments.
Don't have access to unguarded LLMs -> Can't train on constitutions about "bombs"
Select toy example:
- Tarantino
Need not to overfit for "director"
- Generate good data
Need not to overfit to film industry
- Generate neutral data

Val:
- Random samples

Tests:
- On new director
- on new neutral data
- on new attacks
- on new combinations of attacks

Important metrics: (if being tarantino is positive)
- Accuracy 
- TPR (for attack deflection)
- FPR (for benign prompts rejection)

Blubber on why exactly these metrics are important

# Results

- Mimced pipeline on a toy example
- Show graphics

# Discussions



# Conclusion

# References
https://arxiv.org/pdf/2412.02159
https://arxiv.org/pdf/2312.06674
https://arxiv.org/abs/2411.07494

# Contribution: Team description with their roles