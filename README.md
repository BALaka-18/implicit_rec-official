# Implicit_reca

This is an implementation of the ALS algorithm under Apache Spark's mllib library, designed specifically for designing recommendation systems on implicit data, with the recent update on the regularization parameter.

## What is implicit data ?

The standard approach to matrix factorization-based collaborative filtering treats the entries in the user-item matrix as explicit preferences given by the user to the item, for example, users giving ratings to movies.

It is common in many real-world use cases to only have access to implicit feedback (e.g. views, clicks, purchases, likes, shares etc.). This is called implicit data.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install implicit_reca
```

## Usage

```python
import implicit_reca as ir

ir.create_lookup(dataset)                                         # Create the lookup table for future reference.
ir.create_sparse(dataset,name_of_implicit_feautre)                # Create the sparse matrix of user x items (R).
ir.implicit_als(spmx,alpha,iterations,lambd,features)             # Main function behind the ALS algorithm.
ir.item_recommend(item_ID,item_vecs,item_lookup,no_items_rec)     # Item vs item recommendation.
```

## Contributing [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/BALaka-18/implicit_rec-official/issues)


Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
