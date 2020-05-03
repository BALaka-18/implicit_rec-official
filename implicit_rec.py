#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy import nonzero
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import MinMaxScaler

'''LIST OF CONSTITUENT FUNCTIONS :

>> create_lookup(dataset) : Create the lookup table for future reference.

>> create_sparse(dataset,name_of_implicit_feautre) : Create the sparse matrix of user x items (R).

>> implicit_als(spmx,alpha,iterations,lambd,features) : Main function behind the ALS algorithm.

>> item_recommend(item_ID,item_vecs,item_lookup,no_items_rec) : Item vs item recommendation.'''



"""#### STEP 1 : Prepping and Processing the data to the desired (user x item) matrix"""
# Step 1 : Create a lookup to be accessed later
def create_lookup(data):
  '''Function to return the lookup
  
  Parameter:
  data = dataset (must contain the three columns artistID, userID and, Artist_Name)'''
  item_lookup = data[['artistID','userID','Artist_Name']].drop_duplicates()
  item_lookup['artistID'] = item_lookup['artistID'].astype(str)
  return item_lookup


def create_sparse(data,name_of_implicit_feautre):
  # Step 2 : Drop the artist name column
  dat = data.copy()
  dat.drop(columns=['Artist_Name'],inplace=True)

  # Step 3 : Keep rows where weight != 0, as 0 weight means the person hates the artist or hasn't listened to him/her
  dat = dat.loc[dat.name_of_implicit_feautre!=0]

  # Step 4 : Making the sparse matrix
  # Define the dtype of the userID and artistID columns using CategoricalDtype() of pandas. Define ordered relationship as True. 
  user_c = CategoricalDtype(sorted(dat.userID.unique()), ordered=True)     # from dtype int64 to category
  item_c = CategoricalDtype(sorted(dat.artistID.unique()), ordered=True)

  # Define the rows and columns of of the sparse matrix, making sure to use the newly defined dtype inside .astype()
  row = dat.userID.astype(user_c).cat.codes
  col = dat.artistID #.astype(item_c).cat.codes
  spmx = sp.csr_matrix((dat["weight"], (row, col)), \
                            shape=(user_c.categories.size, item_c.categories.size))
  return spmx


'''# Visualizing the sparse matrix with a slightly similar pivot_table
pvt = pd.pivot_table(dat,values='weight',index='userID',columns='artistID')
pvt.fillna(-1,inplace=True)'''


"""#### STEP 2 : Calculating the coefficients, create the X and Y matrices (User and Item vectors respectively)

#### Explaining each of the function arguments and the primary formula behind creating the user and item matrices.
Function name : implicit_als

Params :
1. spmx = sparse matrix, user x items, namely R.
2. alpha = the linear scaling factor, used in calculation of confidence. According to the paper I am taking reference from, best value of alpha = 40.
3. iterations = no. of iterations.
4. lambd = regularization parameter (lambda).
5. features = no. of features we will have in the user vector/factor matrix. 

Returns : X(user x features)(user vector) , Y(features x items)(items vector)

#### R = U X V
  ###### U = User x features
  ###### V = Features x items

### THE MATH :
Explaining the Alternate Least Square mathematically : 

ALS is a method where we find out the *preference(p) for an item* and merge it with the *confidence associated with that item(Cu for user vectors ; Ci for item vectors)* to predict a *recommendation score* that determines its **closeness to a particluar item or its probability of being liked by a particular user.**
________________________________________________________________________________
### CALCULATING PREFERENCE :
p(u) OR p(i) = prefernce, or, the binary representation of our feedback data(here, weight).

**p_ui = 1, if r_ui(weight) > 0 ; else p_ui = 0**
________________________________________________________________________________
### CALCULATING CONFIDENCE :
C_u OR C_i = confidence, or How much a certaing user prefers/likes a certain artist

**Cui = 1 + alpha*r_ui** , where, alpha = linear scaling factor, r_ui = weights.
________________________________________________________________________________

ALS is the same as OLS (Ordinary Least Squares) just that the process of minimizing the loss function alternates between the user and item vectors. *It alternately fixes the user factors or the item factors to calculate the global minimum.*
________________________________________________________________________________
### LOSS FUNCTION MINIMIZATION :

**GOAL :** Minimize loss function

min(summation(u,i)[Cui(Pui + Xu.T.Yi)^(2) + lambd(summation(i)[|x|^2] + summation(u)[|y|^2])])

*WHAT IS LAMBD (LAMBDA) ?* It is the regualrizer to reduce overfitting.

**For user vector** : x_u = (Y.T.Cu.Y + lambd * I)^(-1) * (Y.T * Cu * p(u)) ____(1)

**For item vector** : y_i = (X.T.Ci.X + lambd * I)^(-1) * (X.T * Ci * p(i)) ____(2)

  Now, we can further reduce Cu to Cu = I + (Cu - I) = 1 + (Cu - I) _____(3)

Substituting in (1) and (2) :

**For user vector** : x_u = ((Y.T * Y + Y.T * (Cu - I) * Y) + lambd * I)^(-1) * (Y.T * Cu * p(u))

**For item vector** : y_i = ((X.T * X + X.T * (Ci - I) * X) + lambd * I)^(-1) * (X.T * Ci * p(i))
"""

# STEP 1 : Define a major function, implicit_als, that'll be the main logic behind 
def implicit_als(spmx,alpha=40,iterations=10,lambd=0.1,features=10):

  '''Parameters :
  spmx = Sparse matrix (R -> users x items)
  alpha = linear scaling factor
  iterations = Number of iterations to complete
  lambd = lambda, or, regularization constant
  features = Number of latent features to take into consideration.
  
  Returns : X (the user vector) ; Y (item vector)'''

  # 1. Calculate the confidence for each value
  conf = spmx * alpha     # conf = 1 + alpha*r_ui
  
  # 2. Get the dimensions
  user_size,item_size = spmx.shape

  # 3. Initialize X and Y.
  X = sp.csr_matrix(np.random.normal(size=(user_size,features)))     # User vector : user x features
  Y = sp.csr_matrix(np.random.normal(size=(item_size,features)))     # Item vector : item x features

  # 4. Precompute identity matrices of X and Y
  X_I = sp.eye(user_size)
  Y_I = sp.eye(item_size)

  # 5. Precompute identity matrix of features(I) and addition/regularization term : lambda*I
  I = sp.eye(features)
  lI = lambd * I  # Regularization term

  # ALS Implementation starts here :
  # Iterate as many times mentioned in the function.
  for _ in range(iterations):
    yTy = Y.T.dot(Y)          # Y.T.Y
    xTx = X.T.dot(X)          # X.T.X

    # Loop through all users
    for u in range(user_size):
      # Get confidence values of all artists for that user.
      u_row = conf[u,:].toarray()[0]        # This is Cu

      # Calculate binary prefernce
      p_u = u_row.copy()
      p_u[p_u != 0] = 1.0   # All those with confidence > 0 have preference of 1. Else 0.

      # Calculate Cu and Cu-I
      CuI = sp.diags(u_row, 0)          # 0 defines the main diagonal
      Cu = CuI + Y_I

      # Calculate the confidence based regularized term and the prefernce based terms separately.
      yT_CuI_y = Y.T.dot(CuI).dot(Y)
      yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)

      # Substitute the above calculated terms in the formula of user vector : x_u = ((Y.T * Y + Y.T * (Cu - I) * Y) + lambd * I)^(-1) * (Y.T * Cu * p(u))
      X[u] = linalg.spsolve(yTy + yT_CuI_y + lI,yT_Cu_pu)

      '''How does sp.linalg.spsolve(A,B) work here ? 
      spsolve solves the linear system Ax = B.
      Here, A = yTy + yT_CuI_y + lI
            B = yT_Cu_pu
            x = X[u]
      So, the equation becomes : 
      (yTy + yT_CuI_y + lI) * X[u] = yT_Cu_pu
      => X[u] =  yT_Cu_pu/(yTy + yT_CuI_y + lI)
      => X[u] =  (yTy + yT_CuI_y + lI)^(-1) * yT_Cu_pu
      
      So, we get our desired equation.'''

    # Repeat all we did above, this time for the items, to get the item vector values.
    # Loop through all items
    for i in range(item_size):
      # Get confidence
      i_row = conf[:,i].T.toarray()[0]        # This is Ci.
      # Why the sudden T ? Because item vector much match the condition for matrix multiplication. 
      # As, R = X x Y ; R = (user_size x item_size) ; X = (user_size x features) ; so, Y must be = (features x item_size)

      # Calculate binary prefernce
      p_i = i_row.copy()
      p_i[p_i != 0] = 1.0   # All those with confidence > 0 have preference of 1. Else 0.

      # Calculate Ci and Ci-I
      CiI = sp.diags(i_row, 0)
      Ci = CiI + X_I

      # Substitute in x_u to calculate the user vector from formula
      xT_CiI_x = X.T.dot(CiI).dot(X)
      xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)

      # Substitute to calculate user vector values and place them in the sparse matrix X(user x features)
      Y[i] = linalg.spsolve(xTx + xT_CiI_x + lI,xT_Ci_pi)
  return X,Y

"""#### STEP 3 : Training the Algorithm/model on the training set(spmx, the sparse matrix of weights)"""

'''Step 1 : Train the model using the implicit_als() function
user_vecs,item_vecs = implicit_als(spmx=spmx,alpha=20,iterations=20,features=20)'''

# TRAINING COMPLETE

"""#### STEP 4 : Testing
#### (a) FINDING SIMILAR ITEMS (here, Artists) FROM ITEM VECTOR (ITEM VS ITEM)

**MATHS BEHIND IT :** To calculate similarity scores and find out similar items for a given item_ID, we take the dot product of the item vector and the transpose of the vector corresponding to that ID.

**score** = V.V_i.T

Sample test for : Taylor Swift(51)
"""

def item_recommend(item_ID,item_vecs,item_lookup,no_items_rec=10):
  '''Sample artist ID
  item_ID = 51'''

  '''Parameters :
  item_ID = Item ID of the artist you want to get recommendations for
  item_vecs = Item vector
  item_lookup = The lookup dataframe created for reference of artist names
  no_items_rec = No. of recommendations to fetch
  
  Returns : A dataframe with recommended items and their recommendation scores'''

  # Getting specific item vector for that ID
  item_vec = item_vecs[item_ID].T

  # Calculating similarity score
  scores = item_vecs.dot(item_vec).toarray().reshape(1,-1)[0]
  top_10 = np.argsort(scores)[::-1][:no_items_rec]      # Top 10 similar artists

  # Initilaize empty lists for artist name and similarity/recommendation score
  artists,art_score = [],[]
  for idx in top_10:
    artists.append(item_lookup.Artist_Name.loc[item_lookup.artistID==str(idx)].iloc[0])
    art_score.append(scores[idx])

  similar_items = pd.DataFrame({'artist':artists,'Score':art_score})
  return similar_items