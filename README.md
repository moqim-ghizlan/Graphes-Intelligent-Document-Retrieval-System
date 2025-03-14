# Intelligent Document Retrieval System

## Description
This project aims to develop an advanced document retrieval system that integrates both traditional text-based search methods and graph-based approaches. The objective is to explore how incorporating document relationships within a graph structure can enhance search relevance and ranking.

---

## Objectives
- Implement and compare text-based and graph-enhanced document retrieval techniques.
- Analyze the impact of network structures on information retrieval.
- Evaluate supervised classification models using different feature representations.

---

## Project Structure
- **`./main.ipynb`** : The main file containing the implementation of the project.
- **`./utility.py`** : A utility file containing helper functions used throughout the project.
- **`./models.py`** : Functions for applying different machine learning models.
- **`./graphic.py`** : Functions for graph visualization and network analysis.
- **`./src/data/data_project.csv`** : The dataset used in the project.
- **`./src/files/<file_name>.pdf`** : Documents detailing project requirements and expected results.
- **`./src/tds/<file_name>.py`** : Provided scripts for the project.
- **`./rapport.pdf`** : A report detailing the methodology, results, and conclusions.
- **`./sigma_graph.html`** : An interactive visualization of the constructed graph.
- **`./graph_visualization.svg`** : A static visualization of the graph.

---

## Installation and Requirements

### Step 1: Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/moqim-ghizlan/Intelligent-Document-Retrieval.git
cd Intelligent-Document-Retrieval
```

### Step 2: Install Dependencies
Ensure that Python and the necessary libraries are installed:
```bash
pip install -r requirements.txt
```

### Step 3: Dataset Placement
Ensure that the dataset `data_project.csv` is in the `src/data/` directory. If missing, download it from the project repository.

---

## Execution
Run the main script to execute the document retrieval and classification experiments:
```bash
jupyter notebook main.ipynb
```


---

## Results
The project produces:
- A **text-based search engine** using TF-IDF and Doc2Vec.
- A **graph-enhanced search engine** leveraging document relationships.
- **Clustering and network analysis** to identify structural patterns in the dataset.
- **Supervised classification models** trained with and without graph embeddings.

Generated visualizations:
- **Graph structure analysis** (`sigma_graph.html`, `graph_visualization.svg`).
- **Performance comparison of classifiers** (`classification_comparison.png`).
- **TF-IDF vs. Doc2Vec similarity comparison** (`tfidf_vs_doc2vec.png`).

---

## Google Colab
You can test this project directly on **Google Colab** by clicking [here](https://colab.research.google.com/drive/1Cy7i4txRYcCP5Gv71CLAV9iNnQUDkr7E?usp=sharing).

---

## Authors
- **Moqim Ghizlan**
- **Thibaut Juillard**

---

## License
This project is open-source and licensed under MIT License.
