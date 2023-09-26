# Estudo sobre o banco de dados que será analisado:

## Sobre o banco de dados:

O banco de dados é de uma pesquisa realizada em 2014 que mede as atitudes em ralação a saúde mental e a frequência de transtornos mentais no local de trabalho tecnológico.

O banco de dados está disponível neste [link](https://www.kaggle.com/osmi/mental-health-in-tech-survey).

## Dados presentes na base de dados:

- Timestamp(Registro de data e hora)

- Age (Idade)

- Gender (Genero)

- Country (País)

- state (Se você mora nos Estados Unidos, qual estado ou território)

- self_employed (Você é um trabalhador autônomo?)

- family_history (Você tem histórico familiar de doenças mentais?)

- treatment (Você já procurou tratamento para uma condição de saúde mental?)

- work_interfere (Se você tem uma condição de saúde mental, você acha que isso interfere em seu trabalho?)

- no_employees (O número de funcionários que trabalham na empresa ou organização)

- remote_work (A empresa ou organização onde você trabalha oferece opções de trabalho remoto?)

- tech_company (É uma empresa de tecnologia?)

- benefits (Sua empresa oferece benefícios de saúde mental?)

- care_options (As opções de cuidados de saúde mental fornecidas pelo empregador)

- wellness_program (O empregador oferece um programa de bem-estar que aborda saúde mental?)

- seek_help (O empregador incentiva os funcionários a procurar ajuda para questões de saúde mental?)

- anonymity (A empresa anônima de pesquisa de saúde mental para funcionários?)

- leave (A política de licença da empresa é amigável para pessoas com doenças mentais?)

- mental_health_consequence (Você acha que discutir problemas de saúde mental com seu empregador afetaria negativamente sua carreira?)

- coworkers(Voce estaria disposto a discutir problemas de saúde mental com seus colegas de trabalho?)

- supervisor(Você estaria disposto a discutir problema de saúde mental com seu supervisor direto?)

- mental_health_interview (Você abordaria um problema de saúde mental com um potencial empregador em uma entrevista?)

- phys_health_interview(Você abordaria problemas de saude com um possivel empregador em uam entrevista?)

- mental_vs_physical(Você sente que seu empregado leva saude mental tão seria quanto saúde fisica?)

- obs_consequence(Você já ouviu falar ou observou consequencias negativas dos colegas de trabalho com condições mentais negativas no seu espaço de trabalho?)

- comments(Qualquer comentario ou observação)

## Possiveis perguntas para serem exploradas:

1. Como a frequência de doenças mentais e as atitudes em relação à saúde mental variam de acordo com a localização geográfica?

2. Quais são os indicadores mais fortes de doenças de saúde mental ou de certas atitudes em relação a saúde mental no local de trabalho.

## Observações:

Temos uma base de dados de uma pesquisa sobre saúde mental, temos 27 colunas que representam a quantidade de perguntas feitas na pesquisa, e 1259 linhas que representam a quantidade de pessoas que responderam a pesquisa.

## Objetivo do estudo:

Entender sobre a base de dados e aprimorar os conhecimentos em Python e Data Science, visando um aprendizado em Inteligência Artificial.

## Analise:


```python
import pandas as pd

df = pd.read_csv('survey.csv')
print(df.shape)  # (1259 linhas, 27 colunas)
print(df.head())  # 5 primeiras linhas
```

Dessa forma podemos ver que temos 1259 linhas e 27 colunas, e que as 5 primeiras linhas são as que estão sendo mostradas. Além disso, podemos concluir inicialmente que o genêro com maior frequência é o masculino, e que a idade média é de 32 anos.

Primeira coisa que tentamos fazer é a média de idade das pessoas que responderam a pesquisa.

Analisando o resultado, obtemos uma média com valor muito extremo, o que pode indicar que temos outliers na nossa base de dados, ou seja, valores que estão muito fora do padrão, e que podem atrapalhar a nossa análise.

Fazendo uma breve análise, podemos observar que existem pessoas com idades de 329 e 999 anos, o que é muito fora do padrão, e que pode atrapalhar a nossa análise, então vamos tentar remover esses outliers.

```python

df = df[df['Age'] <= 100]

```

O inverso também é contrario, existem pessoas com idades de 3 e 5 anos, o que também é muito fora do padrão, e que pode atrapalhar a nossa análise, então vamos tentar remover esses outliers.

```python

df = df[(df['Age'] >= 0) & (df['Age'] >= 18)]
    
```

Removendo os outliers, conseguimos obter uma média de aproximadamente 32 anos, o que é um valor mais próximo do real.


```python

print(df['Age'].mean())

```

Agora vamos analisar a frequência de cada idade, para isso vamos utilizar um gráfico de barras.

```python

import matplotlib.pyplot as plt

plt.hist(df['Age'], bins=20, color='blue', alpha=0.7)
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.title('Distribuição de Idades')

plt.show()
```

Majoritariamente observamos que a frequencia das pessoas que responderam a pesquisa estão entre 20 e 40 anos.

Proximo passo é descobrir o genêro que mais respondeu a pesquisa. 

```python
contagem_genero = df['Gender'].value_counts()

print(contagem_genero)
```

Nota-se que algumas pessoas entraram com alguns valores escritos de forma diferente como 'male' e 'Male', que são a mesma coisa, então vamos tentar padronizar esses valores.

```python


# Criar uma função para mapear os gêneros
def mapear_genero(valor):
    valor = str(valor).strip().lower()
    if valor in ['male', 'm', 'man', 'make', 'malr', 'mail', 'maile', 'mal', 'guy (-ish) ^_^', 'enby', 'androgyne', 'male-ish', 'cis male', 'cis man', 'msle', 'trans-female', 'male (cis)', 'nah', 'ostensibly male, unsure what that really means', 'fluid', 'cis male', 'something kinda male?', 'genderqueer']:
        return 'Male'
    elif valor in ['female', 'f', 'woman', 'female (trans)', 'female (cis)', 'femail', 'woman', 'femake', 'cis female', 'queer/she/they', 'non-binary', 'queer', 'neuter', 'agender']:
        return 'Female'
    else:
        return 'Outros'

# Aplicar o mapeamento ao DataFrame
df['Gender'] = df['Gender'].apply(mapear_genero)

# Contagem do número de pessoas por gênero
contagem_genero = df['Gender'].value_counts()

# Exibir a tabela de contagem
contagem_genero
    
```


Agora podemos ver que o genêro que mais respondeu a pesquisa foi o masculino, e que o feminino foi o segundo genêro que mais respondeu a pesquisa.

Agora vamos analisar a frequência de cada genêro, para isso vamos utilizar um gráfico de barras.

```python
contagem_genero.plot(kind='bar', color=['blue', 'red', 'green'])
plt.xlabel('Gênero')
plt.ylabel('Frequência')
plt.title('Distribuição de Gêneros')
plt.show()
```


Majoritariamente observamos que a frequencia das pessoas que responderam a pesquisa são do genêro masculino.

Sabemos até agora então que a idade média das pessoas que responderam a pesquisa é de 32 anos, e que a maioria das pessoas que responderam a pesquisa são do genêro masculino.

### Analise indiviual dos individuos

Inicialmente faz-se a verificação da frequência de individuos que responderam sim para a variavel 'family_history', que representa se o individuo tem histórico familiar de doenças mentais.

```python

contagem_historico_familiar = df['family_history'].value_counts()
contagem_historico_familiar

```

Tivemos um resultado majoritariamente de pessoas que responderam não para a variavel 'family_history'.

Em sequencia fazemos uma verificação da frequência de individuos que responderam sim para a variavel 'treatment', que representa se o individuo já procurou tratamento para uma condição de saúde mental.

```python

contagem_busca_tratamento = df['treatment'].value_counts()
contagem_busca_tratamento

```

Tivemos um resultado majoritariamente de pessoas que responderam sim para a variavel 'treatment'.

Até o momento, temos o seguinte:

-  Boa parte das pessoas que responderam a pesquisa são do genêro masculino.

-  A idade média das pessoas que responderam a pesquisa é de 32 anos.

-  A maioria das pessoas que responderam a pesquisa não tem histórico familiar de doenças mentais.

-  A maioria das pessoas que responderam a pesquisa já procurou tratamento para uma condição de saúde mental.

Faz-se agora uma assimilação entre individuos que responderam 'sim' para a variavel 'family_history' e 'treatment', para verificar se existe alguma relação entre as duas variaveis.

```python

pessoas_com_historico_e_tratamento = df[(df['family_history'] == 'Yes') & (df['treatment'] == 'Yes')]

contagem_pessoas = pessoas_com_historico_e_tratamento.shape[0]

print("Número de pessoas com histórico familiar e que buscaram tratamento:", contagem_pessoas)

```

Tivemos o seguinte resultado: Número de pessoas com histórico familiar e que buscaram tratamento: 362

Agora vamos verificar quantas dessas pessoas são do gênero masculino:

```python

# Filtro para pessoas do sexo masculino com histórico familiar de problemas de saúde mental e que buscaram tratamento
pessoas_do_sexo_masculino_que_buscaram_tratamento_e_tem_historico = df[(df['family_history'] == 'Yes') & (df['treatment'] == 'Yes') & (df['Gender'] == 'Male')]

# Contagem de pessoas do sexo masculino
contagem_pessoas_masculinas = pessoas_do_sexo_masculino_que_buscaram_tratamento_e_tem_historico.shape[0]

# Exibir a contagem
print("Número de pessoas do sexo masculino com histórico familiar e que buscaram tratamento:", contagem_pessoas_masculinas)

```

Obteve-se quase 70% de pessoas do sexo masculino que buscaram tratamento e tem histórico familiar de problemas de saúde mental.

Indo mais afundo e descobrindo a região dessas pessoas:

```python

contagem_pais = df['Country'].value_counts()
contagem_pais

```

Tivemos um resultado majoritariamente de pessoas que responderam a pesquisa são dos Estados Unidos.

Fazendo uma tabela de contigência para verificar o genêro das pessoas que responderam a pesquisa e a região dessas pessoas:

```python

tabela_contingencia = pd.crosstab(df['Country'], df['Gender'])
print(tabela_contingencia)

```

Tivemos um resultado majoritariamente de pessoas que responderam a pesquisa são do genêro masculino e dos Estados Unidos.

Agora vamos fazer uma verificação de quantas pessoas do genêro masculino e de que país são que tem histórico familiar de problemas de saúde mental e que buscaram tratamento:

```python

# Filtro para pessoas com histórico familiar e que buscaram tratamento
pessoas_com_historico_e_tratamento = df[(df['family_history'] == 'Yes') & (df['treatment'] == 'Yes')]


# Tabela de contagem das regiões
contagem_regioes = pessoas_com_historico_e_tratamento['Country'].value_counts()
print(contagem_regioes)

```

Tivemos um resultado majoritariamente de pessoas que responderam a pesquisa são dos Estados Unidos.

Gerando uma interpretação gráfica:

```python

contagem_regioes.plot(kind='bar', color='purple')
plt.xlabel('Região (País)')
plt.ylabel('Número de Pessoas')
plt.title('Distribuição de Pessoas com Histórico Familiar e Tratamento por Região')
plt.show()

```

Sabemos até agora que a pesquisa está concentrada nos Estados Unidos, e que a maioria das pessoas que responderam a pesquisa são do genêro masculino.

Verificando o Estado das pessoas que responderam a pesquisa:

```python

contagem_estado_reside = df['state'].value_counts()
contagem_estado_reside

```

Tivemos um resultado majoritariamente de pessoas que responderam a pesquisa são do estado da Califórnia, Washington e Nova York.


Fazendo uma analise de quantas pessoas do genêro masculino e de que estado são que tem histórico familiar de problemas de saúde mental e que buscaram tratamento:

```python

# Filtro para pessoas com histórico familiar e que buscaram tratamento
pessoas_com_historico_e_tratamento = df[(df['family_history'] == 'Yes') & (df['treatment'] == 'Yes')]


# Tabela de contagem das regiões
contagem_estados = pessoas_com_historico_e_tratamento['state'].value_counts()
print(contagem_estados)

```

Tivemos um resultado majoritariamente de pessoas que responderam a pesquisa são dos estados da Califórnia, Washington e Nova York.

Com isso terminamos a análise individual dos individuos.

Podemos observar que a maioria das pessoas que responderam a pesquisa são do genêro masculino, dos Estados Unidos, do estado da Califórnia, Washington e Nova York, e que tem histórico familiar de problemas de saúde mental e que buscaram tratamento.

### Analise do ambiente de trabalho

Inicia-se verificando se a empresa oferece suporte para saúde mental:

```python

saude_mental_compromete_o_trabalho = df['work_interfere'].value_counts()
saude_mental_compromete_o_trabalho

```

Bom, a maioria das pessoas que responderam afirmaram que a saúde mental  compromete o trabalho.

Agora iremos verificar o tamanho da empresa:

```python
quantidade_de_funcionarios = df['no_employees'].value_counts()
quantidade_de_funcionarios
    
```

Sabemos que a maioria das pessoas que responderam a pesquisa trabalham em empresas de 6-100 funcionários.

Gerando uma tabela de contigência para verificar se existe alguma relação entre as duas variaveis:

```python

# Crie uma tabela de contingência entre as variáveis work_interfere e no_employees
tabela_contingencia = pd.crosstab(df['work_interfere'], df['no_employees'], margins=True, margins_name='Total')

# Exiba a tabela de contingência
print(tabela_contingencia)

```

Observa-se que quanto maior o número de funcionários, menor a frequência de pessoas que responderam que a saúde mental compromete o trabalho.


Verificando agora se a empresa oferece benefícios de saúde mental:

```python

empresa_oferece_beneficio = df['benefits'].value_counts()
empresa_oferece_beneficio

```

A maioria das pessoas que responderam a pesquisa afirmaram que a empresa oferece benefícios de saúde mental.

Agora fazendo uma tabela de contigência para verificar se existe alguma relação entre as duas variaveis:

```python

tabela_contigencia = pd.crosstab(df['benefits'], df['no_employees'], margins=True, margins_name='Total')
tabela_contigencia

```

Quanto maior o número de funcionários, maior a frequência de pessoas que responderam que a empresa oferece benefícios de saúde mental.

Verificando se os individuos se sentem confortáveis em discutir problemas de saúde mental com seus colegas de trabalho:

```python

conversa_saude_com_companheiros = df['coworkers'].value_counts()
conversa_saude_com_companheiros

```

A maioria das pessoas que responderam a pesquisa afirmaram que se sentem confortáveis em discutir problemas de saúde mental com seus colegas de trabalho.

Verificando o genêro das pessoas que responderam a pesquisa e se sentem confortáveis em discutir problemas de saúde mental com seus colegas de trabalho:

```python

genero_masculino_que_conversam_sobre_saude = df[(df['coworkers'].isin(['Yes', 'Some of them'])) & (df['Gender'] == 'Male')]

contagem = genero_masculino_que_conversam_sobre_saude.shape[0]

contagem

```

Cerca de 796 pessoas do genêro masculino se sentem confortáveis em discutir problemas de saúde mental com seus colegas de trabalho.

Temos até agora o seguinte:

-  A maioria das pessoas que responderam a pesquisa são do genêro masculino.

-  A maioria das pessoas que responderam a pesquisa trabalham em empresas de 6-100 funcionários.

-  A maioria das pessoas que responderam a pesquisa afirmaram que a empresa oferece benefícios de saúde mental.

-  A maioria das pessoas que responderam a pesquisa afirmaram que se sentem confortáveis em discutir problemas de saúde mental com seus colegas de trabalho.

- Podemos analisar até então uma quebra de estigma em relação a saúde mental e o genêro masculino, pois a maioria das pessoas que responderam a pesquisa são do genêro masculino, e a maioria dessas pessoas afirmaram que se sentem confortáveis em discutir problemas de saúde mental com seus colegas de trabalho.

Verificando se as pessoas aham que a saúde fisica e tão importante quanto a saúde mental:

```python

saude_fisica_x_mental = df['mental_vs_physical'].value_counts()
saude_fisica_x_mental

```

As resposta foram bem divididas, mas a maioria das pessoas que responderam a pesquisa afirmaram que a saúde mental é tão importante quanto a saúde física.

Verificando se a questão da saúde mental seria abordada em uma entrevista de emprego:

```python

saude_mental_na_entrevista = df['phys_health_interview'].value_counts()
saude_mental_na_entrevista

```

A resposta foi bem dividida, mas a maioria das pessoas que responderam a pesquisa afirmaram que a questão da saúde mental não seria abordada em uma entrevista de emprego.

Verificando se o tamanho da empresa influencia essa resposta:

```python

tabela_contigencia = pd.crosstab(df['phys_health_interview'], df['no_employees'], margins=True, margins_name='Total')
tabela_contigencia

```

Aqui temos uma resposta curiosa, quanto maior a empresa menores são as chances de a questão da saúde mental ser abordada em uma entrevista de emprego, porém maiores são as chances deles oferecem benefícios de saúde mental.

Já as pequenas empresas, tem maiores chances de abordar a questão da saúde mental em uma entrevista de emprego, porém menores são as chances de oferecerem benefícios de saúde mental.


### Conclusão

Com base na análise realizada dos dados da pesquisa sobre saúde mental no local de trabalho em 2014, podemos tirar algumas conclusões e observações importantes:

1. **Idade e Gênero**: A maioria dos respondentes da pesquisa era do gênero masculino, com uma idade média de cerca de 32 anos.

2. **Histórico Familiar e Busca de Tratamento**: A maioria dos participantes não tinha histórico familiar de doenças mentais, mas muitos deles buscaram tratamento para problemas de saúde mental. Isso sugere que as doenças mentais podem ser percebidas de maneira independente do histórico familiar.

3. **Localização Geográfica**: A pesquisa foi realizada principalmente nos Estados Unidos, com uma concentração significativa na Califórnia, Washington e Nova York. Portanto, as conclusões podem se aplicar principalmente a esse contexto geográfico.

4. **Ambiente de Trabalho**: A maioria dos participantes relatou que a saúde mental afetou seu trabalho. No entanto, as empresas de maior porte parecem estar menos dispostas a abordar a questão da saúde mental em entrevistas de emprego, embora sejam mais propensas a oferecer benefícios de saúde mental. 

5. **Quebra de Estigma**: É notável que a maioria dos respondentes, principalmente do gênero masculino, afirmou se sentir confortável discutindo problemas de saúde mental com seus colegas de trabalho. Isso pode refletir uma crescente quebra de estigma em relação à saúde mental no local de trabalho, especialmente em um setor como a tecnologia.

6. **Importância da Saúde Mental**: A maioria dos respondentes concordou que a saúde mental é tão importante quanto a saúde física, destacando um aumento na conscientização sobre a importância da saúde mental.

