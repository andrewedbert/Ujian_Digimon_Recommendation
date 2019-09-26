'''
========================================================================================================================
Soal 3 - Pokemon Battle
========================================================================================================================
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
========================================================================================================================
Read CSV Data
========================================================================================================================
'''
pokemon = pd.read_csv('pokemon.csv')
pokemon = pokemon.drop(['Type 1','Type 2','Generation','Legendary'],axis = 1)

combats = pd.read_csv('combats.csv')

# columnPokemon = list(pokemon.columns)
# print(columnPokemon)

# columnCombat = list(combats.columns)
# print(columnCombat)

'''
========================================================================================================================
Check null value
========================================================================================================================
'''
# print(pokemon.isnull().sum())

'''
========================================================================================================================
Joining Dataframe
========================================================================================================================
'''
pokemon['total'] = pokemon['HP'] + pokemon['Attack'] + pokemon['Defense'] + pokemon['Sp. Atk'] + pokemon['Sp. Def'] + pokemon['Speed']

name_dict = dict(zip(pokemon['#'], pokemon['Name']))
hp_dict = dict(zip(pokemon['#'], pokemon['HP']))
attack_dict = dict(zip(pokemon['#'], pokemon['Attack']))
defense_dict = dict(zip(pokemon['#'], pokemon['Defense']))
spattack_dict = dict(zip(pokemon['#'], pokemon['Sp. Atk']))
spdefense_dict = dict(zip(pokemon['#'], pokemon['Sp. Def']))
speed_dict = dict(zip(pokemon['#'], pokemon['Speed']))
total_dict = dict(zip(pokemon['#'], pokemon['total']))

# print(pokemon.head())

# pokemon1 = 'Bulbasaur'
# pokemon2 = 'Pikachu'

# idPokemon1 = pokemon['#'][pokemon['Name']== pokemon1].values[0]
# idPokemon2 = pokemon['#'][pokemon['Name']== pokemon2].values[0]
# print(idPokemon1)
# print(idPokemon2)

df_combat = combats.copy()

df_combat['First_pokemon_name'] = df_combat['First_pokemon'].replace(name_dict)
df_combat['First_pokemon_hp'] = df_combat['First_pokemon'].replace(hp_dict)
df_combat['First_pokemon_attack'] = df_combat['First_pokemon'].replace(attack_dict)
df_combat['First_pokemon_defense'] = df_combat['First_pokemon'].replace(defense_dict)
df_combat['First_pokemon_spattack'] = df_combat['First_pokemon'].replace(spattack_dict)
df_combat['First_pokemon_spdefense'] = df_combat['First_pokemon'].replace(spdefense_dict)
df_combat['First_pokemon_speed'] = df_combat['First_pokemon'].replace(speed_dict)
df_combat['First_pokemon_total'] = df_combat['First_pokemon'].replace(total_dict)

df_combat['Second_pokemon_name'] = df_combat['Second_pokemon'].replace(name_dict)
df_combat['Second_pokemon_hp'] = df_combat['Second_pokemon'].replace(hp_dict)
df_combat['Second_pokemon_attack'] = df_combat['Second_pokemon'].replace(attack_dict)
df_combat['Second_pokemon_defense'] = df_combat['Second_pokemon'].replace(defense_dict)
df_combat['Second_pokemon_spattack'] = df_combat['Second_pokemon'].replace(spattack_dict)
df_combat['Second_pokemon_spdefense'] = df_combat['Second_pokemon'].replace(spdefense_dict)
df_combat['Second_pokemon_speed'] = df_combat['Second_pokemon'].replace(speed_dict)
df_combat['Second_pokemon_total'] = df_combat['Second_pokemon'].replace(total_dict)

df_combat['First_win'] = df_combat.apply(
    lambda col: 1 if col['Winner'] == col['First_pokemon'] else 0, axis=1
)

X = df_combat.drop(['First_pokemon', 'First_pokemon_name', 'Second_pokemon', 'Second_pokemon_name', 'Winner', 'First_win'], axis=1)
Y = df_combat['First_win']

'''
===============================================================================================
Split the data using sklearn module
===============================================================================================
'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y, 
    test_size=0.3, 
    random_state=101
)

'''
===============================================================================================
Machine Learning
===============================================================================================
'''
from sklearn.ensemble import RandomForestClassifier
randomFor = RandomForestClassifier(n_estimators=100)
randomFor.fit(X_train, Y_train)

from sklearn.linear_model import LogisticRegression
logRes = LogisticRegression(solver='lbfgs')
logRes.fit(X_train, Y_train)
'''
===============================================================================================
Save Model
===============================================================================================
'''
import joblib
joblib.dump(randomFor, 'MLmodel')
joblib.dump(logRes, 'MLmodelLog')
