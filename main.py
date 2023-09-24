import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('survey.csv')
print(df.shape)  # (1259 linhas, 27 colunas)
print(df.head())  # 5 primeiras linhas

media = df['Age'].mean()
print(media)

# Classificar o DataFrame com base na coluna 'Age' em ordem decrescente
df_sorted = df.sort_values(by='Age', ascending=False)

# Exibir as 5 maiores idades
maiores_idades = df_sorted.head(5)
print(maiores_idades)

# Observe que existem duas pessoas que colocaram valores exorbitantes

# Excluindo entradas com idades exageradas (por exemplo, acima de 100)
df = df[df['Age'] <= 100]


#  Classificar o DataFrame com base na coluna 'Age' em ordem crescente
df_sorted = df.sort_values(by='Age', ascending=True)

# Exibir as 5 menores idades
menores_idades = df_sorted.head(5)
print(menores_idades)

# Observe que existem pessoas que colocaram valores negativos e baixos para
# o campo Idade, sendo que a idade não pode ser negativa e por se tratar de uma
# pesquisa com pessoas adultas, não pode ser menor que 18 anos.

# Excluindo entradas com idades negativas e menores que 18 anos

df = df[(df['Age'] >= 0) & (df['Age'] >= 18)]

# Calculando a média para verificar se os valores foram normalizados:

media = df['Age'].mean()
print(media)

# A média aparenta estar com um valor mais próximo da realidade. Obtemos um
# valor de aproximadamente 32 anos!


# Plote um histograma das idades
plt.hist(df['Age'], bins=20, color='blue', alpha=0.7)
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.title('Distribuição de Idades')

# Exiba o gráfico
plt.show()

# Podemos observar que a maior parte das pessoas que responderam a pesquisa
# tem entre 20 e 40 anos de idade.

# Contagem do número de pessoas por gênero
contagem_genero = df['Gender'].value_counts()

# Exibir a tabela de contagem
print(contagem_genero)


# Criar uma função para mapear os gêneros
def mapear_genero(valor):
    valor = str(valor).strip().lower()
    if valor in [
            'male', 'm', 'man', 'make', 'malr', 'mail', 'maile', 'mal',
            'guy (-ish) ^_^', 'enby', 'androgyne', 'male-ish', 'cis male',
            'cis man', 'msle', 'trans-female', 'male (cis)', 'nah',
            'ostensibly male, unsure what that really means', 'fluid',
            'cis male', 'something kinda male?', 'genderqueer']:

        return 'Male'

    elif valor in [
            'female', 'f', 'woman', 'female (trans)', 'female (cis)',
            'femail', 'woman', 'femake', 'cis female', 'queer/she/they',
            'non-binary', 'queer', 'neuter', 'agender']:

        return 'Female'

    return 'Outros'


# Aplicar o mapeamento ao DataFrame
df['Gender'] = df['Gender'].apply(mapear_genero)

# Contagem do número de pessoas por gênero
contagem_genero = df['Gender'].value_counts()

# Exibir a tabela de contagem
print(contagem_genero)

# Plote um gráfico de barras com a contagem de gêneros

contagem_genero.plot(kind='bar', color=['blue', 'red', 'green'])
plt.xlabel('Gênero')
plt.ylabel('Frequência')
plt.title('Distribuição de Gêneros')
plt.show()

# Podemos observar que a maior parte das pessoas que responderam a pesquisa
# são do gênero masculino.

# Contagem do número de pessoas por pais
contagem_pais = df['Country'].value_counts()
print(contagem_pais)
