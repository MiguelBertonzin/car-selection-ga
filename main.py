import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt

# Carrega o CSV corrigido
df = pd.read_csv("fipe_corrigido.csv")

# Seleciona colunas relevantes e remove valores ausentes
df = df[['brand', 'model', 'fuel', 'gear', 'engine_size', 'year_model', 'avg_price_brl', 'age_years']]
df = df.dropna()

# Armazena marca original
df['original_brand'] = df['brand']

# Normaliza colunas numéricas
scaler = MinMaxScaler()
df[['engine_size', 'year_model', 'avg_price_brl', 'age_years']] = scaler.fit_transform(
    df[['engine_size', 'year_model', 'avg_price_brl', 'age_years']]
)

# One-hot encoding das categorias
df = pd.get_dummies(df, columns=['brand', 'fuel', 'gear'])

# Define características
numerical_features = ['engine_size', 'year_model', 'avg_price_brl', 'age_years']
dummies_cols = [col for col in df.columns if col not in ['model', 'engine_size', 'year_model', 'avg_price_brl', 'age_years', 'original_brand']]
num_genes = len(numerical_features) + len(dummies_cols)

# Fitness: quanto maior, melhor
def fitness(chromosome):
    weights = np.array(chromosome)
    score = (
        weights[0] * df['engine_size'] +
        weights[1] * df['year_model'] +
        weights[2] * (1 - df['avg_price_brl']) +
        weights[3] * (1 - df['age_years'])
    )
    for i, col in enumerate(dummies_cols):
        score += weights[4 + i] * df[col]
    return score.mean()

# Inicializa população
def initialize_population(size):
    return [np.random.rand(num_genes).tolist() for _ in range(size)]

# Seleção dos melhores
def selection(population, scores, num_best):
    top_indices = np.argsort(scores)[-num_best:]
    return [population[i] for i in top_indices]

# Crossover simples
def crossover(p1, p2):
    point = random.randint(1, num_genes - 2)
    return p1[:point] + p2[point:]

# Mutação
def mutate(chrom, mutation_rate=0.1):
    for i in range(len(chrom)):
        if random.random() < mutation_rate:
            chrom[i] = np.clip(chrom[i] + np.random.normal(0, 0.1), 0, 1)
    return chrom

# Executa o AG
def run_genetic_algorithm(generations=30, pop_size=40, elite_size=6):
    population = initialize_population(pop_size)
    best_scores = []

    for gen in range(generations):
        scores = [fitness(ind) for ind in population]
        best_scores.append(max(scores))
        elites = selection(population, scores, elite_size)
        next_gen = elites.copy()

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)

        population = next_gen

    final_scores = [fitness(ind) for ind in population]
    best_index = np.argmax(final_scores)
    return population[best_index], best_scores

# Executa o algoritmo genético
best_chromosome, score_history = run_genetic_algorithm()

# Mostra a evolução do fitness
plt.plot(score_history, marker='o')
plt.title("Evolução do Score Médio por Geração")
plt.xlabel("Geração")
plt.ylabel("Fitness (Score Médio)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Exibe o melhor cromossomo encontrado
print("\nMelhor vetor de preferências encontrado pelo AG:")
print(np.round(best_chromosome[:4], 2))