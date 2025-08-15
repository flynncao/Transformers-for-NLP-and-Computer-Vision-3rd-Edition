
# # The Attention Mechanism
# Copyright 2021-2024, Denis Rothman, MIT License. Denis Rothman rewrote the reference notebook entirely in basic Python with no frameworks. Three more steps were added, and a Hugging Face transformer example was added. The original images were taken out, redesigned by Denis Rothman for educational purposes, and inserted in the book descriptions of the multi-attention sub-layer.
# 
# The goal of this notebook is to obtain a mathematical view of the attention mechanism of transformer models. An industry 4.0 developer will become and AI expert with in-depth NLP knowledge.
# 
# **June 10,2024 update** The typos in the Figures for Steps 6 and 7 have been fixed.
# 
# **Note:** In some instances, the actual numerical values include many decimal places. To simplify visualization, some figures in the notebook display rounded values for the attention heads' flow.
# 
# [The Reference Colaboratory Notebook was written by Manuel Romero](https://colab.research.google.com/drive/1rPk3ohrmVclqhH7uQ7qys4oznDdAhpzF)
# 
# [A Medium article was written by Raimi Karim](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)


# #Step 1: Represent the input


from IPython.display import Image     #This is used for rendering images in the notebook


import requests
from PIL import Image
from io import BytesIO

def get_image_from_github(image_name):
    # The base URL of the image files in the GitHub repository
    base_url = 'https://raw.githubusercontent.com/Denis2054/Transformers-for-NLP-and-Computer-Vision-3rd-Edition/main/Notebook%20images/02/'

    # Make the request
    response = requests.get(base_url + image_name)

    # Check if the request was successful
    if response.status_code == 200:
        # Read the image
        image = Image.open(BytesIO(response.content))

        # Return the image
        return image
    else:
        print(f'Error {response.status_code}: Could not access the image file.')
        return None


import numpy as np
from scipy.special import softmax


print("Step 1: Input : 3 inputs, d_model=4") # 3 tokens, d_model=4, so (3, 4) matrix
x =np.array([[1.0, 0.0, 1.0, 0.0],   # Input 1
             [0.0, 2.0, 0.0, 2.0],   # Input 2
             [1.0, 1.0, 1.0, 1.0]])  # Input 3
print(x)


display(get_image_from_github('B19899_02_11.png') or print('Image not found'))


# #Step 2: Initializing the weight matrices


print("Step 2: weights 3 dimensions x d_model=4") # weight matrices should be (4,4)
print("w_query")
w_query =np.array([[1, 0, 1],
                   [1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 1]])
print(w_query)


print("w_key")
w_key =np.array([[0, 0, 1],
                 [1, 1, 0],
                 [0, 1, 0],
                 [1, 1, 0]])
print(w_key)


print("w_value")
w_value = np.array([[0, 2, 0],
                    [0, 3, 0],
                    [1, 0, 3],
                    [1, 1, 0]])
print(w_value)


display(get_image_from_github('B19899_02_12.png') or print('Image not found'))


# #Step 3: Matrix multiplication to obtain Q, K, and V


print("Step 3: Matrix multiplication to obtain Q,K,V")
print(x)
print(f'w_query\n{w_query}')
print("Queries: x * w_query")
Q=np.matmul(x,w_query)
print(Q)


print("Step 3: Matrix multiplication to obtain Q,K,V")

print("Keys: x * w_key")
K=np.matmul(x,w_key)
print(K)


print("Values: x * w_value")
V=np.matmul(x,w_value)
print(V)


display(get_image_from_github('B19899_02_13.png') or print('Image not found'))


# #Step 4: Scaled attention scores


print("Step 4: Scaled Attention Scores")
k_d=1   #square root of k_d simplified to 1 for this example
attention_scores = (Q @ K.transpose())/k_d # @ is the same as np.matmul
print(attention_scores)


display(get_image_from_github('B19899_02_14.png') or print('Image not found'))


# #Step 5: Scaled softmax attention scores for each vector


print("Step 5: Scaled softmax attention_scores for each vector")
attention_scores[0]=softmax(attention_scores[0])
attention_scores[1]=softmax(attention_scores[1])
attention_scores[2]=softmax(attention_scores[2])
print(attention_scores[0])
print(attention_scores[1])
print(attention_scores[2])


display(get_image_from_github('B19899_02_15.png') or print('Image not found'))


# #Step 6: The final attention representations


print("Step 6: attention value obtained by score1/k_d * V")
print(V[0])
print(V[1])
print(V[2])
print("Attention 1")
attention1=attention_scores[0].reshape(-1,1)
attention1=attention_scores[0][0]*V[0]
print(attention1)

print("Attention 2")
attention2=attention_scores[0][1]*V[1]
print(attention2)

print("Attention 3")
attention3=attention_scores[0][2]*V[2]
print(attention3)


display(get_image_from_github('B19899_02_16.png') or print('Image not found'))


# #Step 7: Summing up the results


print("Step 7: summed the results to create the first line of the output matrix")
attention_input1=attention1+attention2+attention3
print(attention_input1)


display(get_image_from_github('B19899_02_17.png') or print('Image not found'))


# #Step 8: Steps 1 to 7 for all the inputs


print("Step 8: Step 1 to 7 for inputs 1 to 3")
#We assume we have 3 results with learned weights (they were not trained in this example)
#We assume we are implementing the original Transformer paper. We will have 3 results of 64 dimensions each
attention_head1=np.random.random((3, 64))
print(attention_head1)


# #Step 9: The output of the heads of the attention sublayer


print("Step 9: We assume we have trained the 8 heads of the attention sub-layer")
z0h1=np.random.random((3, 64))
z1h2=np.random.random((3, 64))
z2h3=np.random.random((3, 64))
z3h4=np.random.random((3, 64))
z4h5=np.random.random((3, 64))
z5h6=np.random.random((3, 64))
z6h7=np.random.random((3, 64))
z7h8=np.random.random((3, 64))
print("shape of one head",z0h1.shape,"dimension of 8 heads",64*8)


# #Step 10: Concatenation of the output of the heads


print("Step 10: Concatenation of heads 1 to 8 to obtain the original 8x64=512 output dimension of the model")
output_attention=np.hstack((z0h1,z1h2,z2h3,z3h4,z4h5,z5h6,z6h7,z7h8))
print(output_attention)


display(get_image_from_github('B19899_02_18.png') or print('Image not found'))


display(get_image_from_github('B19899_02_19.png') or print('Image not found'))


