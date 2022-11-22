import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report


# Puxando os dados do csv para o data frame com pandas
def entrada_df():
	df = pd.read_csv('games.csv')
	return df

# Capturando os cenários para analises, os cenários estão localizados nas linhas 0 a 2
def captura_cenarios(df):
	cenarios = df[0:3]
	df.drop(3, axis = 0, inplace=True)
	return cenarios

#Pre processamento: renomeando colunas, criando coluna relevante, 
def funcao_pre_processamento(df):
	colunasusadas = ['rated','turns','victory_status','increment_code','white_rating','black_rating',
		'opening_eco','opening_ply', 'opening_name','winner'] 	
	df_filtrado = df.loc[: , colunasusadas]
	df_filtrado.rename(columns = {'rated':'ranqueada', 'turns':'turnos', 'victory_status':'finalizacao_partida', 'increment_code':'incremento_turno', 'white_rating':'ranque_brancas',
									'black_rating':'ranque_pretas', 'opening_eco':'abertura_codigo', 'opening_ply':'movimentos_abertura','opening_name':'nome_abertura', 'winner':'ganhador'}, inplace = True)
    
	df_filtrado['diferenca_pontos'] = df_filtrado['ranque_brancas'] - df_filtrado['ranque_pretas']
	colunas_trans_ordinais = ['finalizacao_partida','ganhador','incremento_turno','abertura_codigo','nome_abertura']
	le = LabelEncoder()
	df_filtrado[colunas_trans_ordinais] = df_filtrado[colunas_trans_ordinais].apply(le.fit_transform)
	cenarios = captura_cenarios(df_filtrado)
	x = df_filtrado[df_filtrado.columns.difference(['ganhador'])]
	y = df_filtrado['ganhador']

	return x, y, cenarios 

def funcao_split_treino(x, y):
	x_treino, teste_entrada_X, treino_saida_y, test_saida_y = train_test_split(x, y,
                 test_size = 0.30, random_state = 101)
	sm = SMOTE(random_state = 101)
	x_treino, treino_saida_y = sm.fit_resample(x_treino,treino_saida_y)
	minmax = MinMaxScaler()
	x_treino = minmax.fit_transform(x_treino)
	teste_entrada_X = minmax.fit_transform(teste_entrada_X)
	return x_treino, teste_entrada_X, treino_saida_y, test_saida_y

def randomRandomForest(treino_entrada_X, treino_saida_Y):
	rf_model = RandomForestClassifier()
	parametro_modelo = {
		
		# números de amostra aleatória de 4 a 204
		'n_estimators': [4, 200],
		# max_fetures normalmente distribuídos, com média 0,25 stddev 0,1, limitado entre 0 e 1
		'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
		# distrubuição uniforme de 0.01 a 0.2 (0.01 + 0.199)
		'min_samples_split': [2 ,5 ,10],
		'max_depth':[10, 50, 100],
	}
	randomsearch_forest = RandomizedSearchCV(rf_model, parametro_modelo, n_iter=100, cv=5, random_state=1, scoring = 'accuracy')
	randomsearch_forest.fit(treino_entrada_X, treino_saida_Y)

	print("O melhor parametro escolhido para o Random Forest é: ")
	print(randomsearch_forest.best_params_)

	return randomsearch_forest

def gridKNearestN(treino_entrada_X, treino_saida_Y):
	param_grid_knn = [
					{'n_neighbors': [1, 5, 10, 25, 50, 750], 'weights': ['distance','uniform'],'leaf_size': [1, 5, 10, 25, 50, 750], 
					 'p':[1, 2]},
	]
	searchgrid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, verbose = 3, scoring = 'accuracy')
	searchgrid_knn.fit(treino_entrada_X, treino_saida_Y)

	print("O melhor parametro escolhido para o kNN é: ")
	print(searchgrid_knn.best_params_)

	return searchgrid_knn


def menu():   
	loop = True	
	#Chamada do Dataframe para var 'df'
	df = entrada_df()
	#Chamada de pre-processamento (filtragem, acrescentar nova coluna e etc...)
	x, y, cenarios = funcao_pre_processamento(df)
	#Chamda de split do dataset para treinamento e teste
	treino_entrada_X, teste_entrada_X, treino_saida_Y, test_saida_y = funcao_split_treino(x, y)

	print("\nDefina o modelo de aprendizagem:")  
	while loop:
		print("\n Digite [1] para que o modelo escolha aprendizagem por KNN com hiper-parametrização Grid Seach")
		print("\n Digite [2] para que o modelo escolha aprendizagem por Random-Forest com hiper-parametrização Random Seach")
		print("\n Digite [3] para fechar o código \n")
		escolha = input("Escolha dentre as opções anteriores: ")

		if escolha == '1':

			#Analise do modelo em geral
			knn_predicao = gridKNearestN(treino_entrada_X, treino_saida_Y)
			print(classification_report(test_saida_y, knn_predicao.predict(teste_entrada_X)))

			#Analise para o cenário com knn
			# pred_cenario = knn_predicao.predict(cenarios.drop('ganhador', axis=1))
			# pred_prob_cenario = knn_predicao.predict_proba(cenarios.drop('ganhador', axis=1))
			# print("Predizer a probabilidade:")
			# print(pred_prob_cenario)
			# print("Predizer o ganhador:")
			# print(pred_cenario)

		elif escolha == '2':

			#Analise do modelo em geral
			random_forest_predicao = randomRandomForest(treino_entrada_X, treino_saida_Y)
			print(classification_report(test_saida_y, random_forest_predicao.predict(teste_entrada_X)))			


			#Analise para o cenário com random forest
			# pred_cenario = random_forest_predicao.predict(cenarios.drop('ganhador', axis=1))
			# pred_prob_cenario = random_forest_predicao.predict_proba(cenarios.drop('ganhador', axis=1))
			# print("Predizer a probabilidade:")
			# print(pred_prob_cenario)
			# print("Predizer o ganhador:")
			# print(pred_cenario)


		elif escolha == '3':
			return  

		else:
			print("Escolha inválida, as opções são 1, 2 ou 3.")


menu()


# df = entrada_df()
# cenarios = captura_cenarios(df)

# print(cenarios)

