# Exercises of Jax, Inputs via grain's DatasetIterator.

Start from this set of exercises, the whole flow will be more integrated as a whole.
(In contrast, [earlier exercises](jax-only.md) treat each testcase a separate run.)

## CSES-3419 (https://cses.fi/problemset/task/3419) (nimber calculation)

First try of the setup. A custom grain Dataset will collect all testcases.

[Whole program](cses-3419/whole.py). Everything squeezed into a single file for now.
And the dataset is created by globbing a directory and getting all text input files, each file as one datapoint.

Can I convert the mechanism to use `DataLoader` instead?
