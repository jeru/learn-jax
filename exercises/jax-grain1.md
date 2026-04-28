# Exercises of Jax, Inputs via grain's DatasetIterator.

Start from this set of exercises, the whole flow will be more integrated as a whole.
(In contrast, [earlier exercises](jax-only.md) treat each testcase a separate run.)

## CSES-3419 (https://cses.fi/problemset/task/3419) (nimber calculation)

First try of the setup. A custom grain `Dataset` collects all testcases.

[Whole program](cses-3419/whole.py). Everything squeezed into a single file for now.
And the dataset is created by globbing a directory and getting all text input files, each file as one datapoint.

Can I convert the mechanism to use `DataLoader` instead?

## CSES-1621 (https://cses.fi/problemset/task/1621) (size(unique(...)))

Make this input as a `DataSource` to use the higher-level, fancier grain tooling (DataLoader). Need to set a `num_epochs` so it doesn't loop forever.

[Whole program](cses-1621/whole.py).

## CSES-1619 (https://cses.fi/problemset/task/1619) (interval overlapping)

Make the DataSource a [separate library](../src/my_lib/tests_data_source.py) for reuse.

[Whole program](cses-1619/whole.py).

## CSES-1640 (https://cses.fi/problemset/task/1640) (two values from array)

Now move to use grain Transformation for input data processing.

[Whole program](cses-1640/whole.py).
Noticeably now `grain.load` is also supplied a list of transformations.
