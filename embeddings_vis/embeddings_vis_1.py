import re
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_product_info(markdown_file):
    with open(markdown_file, 'r') as file:
        content = file.read()

    pattern = r'## (.+?)\s+(— .+?)(?=\n## |\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    product_info = []
    for match in matches:
        product_name = match[0].strip()
        product_description = match[1].strip()
        product_info.append((product_name, product_description))

    return product_info

def generate_embeddings(product_info, model):
    product_names = [info[0] for info in product_info]
    product_descriptions = [info[1] for info in product_info]

    embeddings = model.encode(product_descriptions)
    return product_names, embeddings

def save_embeddings(product_names, embeddings, output_file):
    np.save(output_file, embeddings)

def save_product_info_to_csv(product_info, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Product Name', 'Product Description'])
        writer.writerows(product_info)

def plot_embeddings_3d(product_names, embeddings):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = embeddings[:, 0]
    y = embeddings[:, 1]
    z = embeddings[:, 2]

    ax.scatter(x, y, z)

    for i, name in enumerate(product_names):
        ax.text(x[i], y[i], z[i], name, fontsize=8)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('Product Embeddings in 3D Space')

    plt.tight_layout()
    plt.show()

# Set up the SentenceTransformer model
model = SentenceTransformer("infgrad/stella_en_1.5B_v5", trust_remote_code=True).cuda()

# Extract product information from the markdown file
markdown_file = 'product_info.md'
product_info = extract_product_info(markdown_file)

# Generate embeddings for product descriptions
product_names, embeddings = generate_embeddings(product_info, model)

# Save the embeddings to a file
output_file = 'product_embeddings.npy'
save_embeddings(product_names, embeddings, output_file)

# Save the product information to a CSV file
csv_file = 'product_info.csv'
save_product_info_to_csv(product_info, csv_file)

# Plot the embeddings in 3D space
plot_embeddings_3d(product_names, embeddings)