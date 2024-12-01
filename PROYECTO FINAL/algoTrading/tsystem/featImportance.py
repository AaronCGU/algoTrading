import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score # Nuestra metrica... 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Función lambda externa para el criterio de selección (top 10)
selectionCriteria = lambda df: df.nlargest(10, 'Importance')['Feature']

# Clase Principal - Feature Importance
class FeatImp(object):
    """
    Clase para calcular la importancia de las características utilizando diferentes métodos
    y opcionalmente seleccionar las principales características.
    """
    
    def __init__(self, method='mdi', automatize=False, labels_return=False, nestimators=200, max_iter=2000, rs=42):
        """
        Inicializa la clase FeatImp.
        
        Parámetros:
        - method (str): 'mdi' para Mean Decrease Impurity o 'mda' para Mean Decrease Accuracy.
        - automatize (bool): Si es True, selecciona las 5 principales características basadas en la importancia.
        - labels_return (bool): Si es True, retorna las etiquetas junto con las características.
        - nestimators (int): Número de árboles para RandomForestClassifier (usado en 'mdi').
        - max_iter (int): Número máximo de iteraciones para LogisticRegression (usado en 'mda').
        - rs (int): Semilla para reproducibilidad.
        """
        self.method = method
        self.automatize = automatize
        self.labels_return = labels_return
        self.nestimators = nestimators
        self.max_iter = max_iter
        self.rs = rs
        self.selected_features = None
        self.feat_importances = None
    
    def feat_imp_mdi(self, target, features):
        """
        Calcula la importancia de las características usando Mean Decrease Impurity.
        
        Parámetros:
        - target: Etiquetas objetivo.
        - features: Matriz de características.
        
        Retorna:
        - DataFrame con las características, su importancia y ranking.
        """
        # Ajusta el modelo Random Forest
        rf = RandomForestClassifier(n_estimators=self.nestimators, random_state=self.rs)
        rf.fit(features, target)

        # Obtiene las importancias de las características
        importances = rf.feature_importances_

        # Crea un DataFrame para las importancias
        feat_importances = pd.DataFrame({
            'Feature': features.columns,
            'Importance': importances
        })
        feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feat_importances['Rank'] = feat_importances['Importance'].rank(ascending=False)

        return feat_importances
    
    def feat_imp_mda(self, target, features):
        """
        Calcula la importancia de las características usando Mean Decrease Accuracy mediante permutación.
        
        Parámetros:
        - target: Etiquetas objetivo.
        - features: Matriz de características.
        
        Retorna:
        - DataFrame con las características, su importancia y ranking.
        """
        # Divide los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=self.rs)

        # Ajusta el modelo de regresión logística
        clf = LogisticRegression(max_iter=self.max_iter, random_state=self.rs)
        clf.fit(X_train, y_train)

        # Precisión base
        baseline_score = accuracy_score(y_test, clf.predict(X_test))

        # Inicializa el diccionario de importancias
        importances = {}

        # Para cada característica
        for col in X_test.columns:
            X_test_permuted = X_test.copy()
            # Permuta la característica
            X_test_permuted[col] = np.random.permutation(X_test_permuted[col])
            # Calcula la nueva precisión
            score = accuracy_score(y_test, clf.predict(X_test_permuted))
            # La importancia es la disminución en precisión
            importances[col] = baseline_score - score

        # Crea un DataFrame
        feat_importances = pd.DataFrame.from_dict(importances, orient='index', columns=['Importance'])
        feat_importances.index.name = 'Feature'
        feat_importances.reset_index(inplace=True)
        feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feat_importances['Rank'] = feat_importances['Importance'].rank(ascending=False)

        return feat_importances
    
    def fit(self, target, features):
        """
        Ajusta el modelo de importancia de características basado en el método elegido.
        
        Parámetros:
        - target: Etiquetas objetivo.
        - features: Matriz de características.
        """
        if self.method == 'mdi':
            self.feat_importances = self.feat_imp_mdi(target, features)
        elif self.method == 'mda':
            self.feat_importances = self.feat_imp_mda(target, features)
        else:
            raise ValueError("El parámetro 'method' debe ser 'mdi' o 'mda'")
        
        if self.automatize:
            # Selecciona las 10 principales características usando selectionCriteria
            self.selected_features = selectionCriteria(self.feat_importances).tolist()
        else:
            self.selected_features = features.columns.tolist()
    
    def transform(self, features):
        """
        Transforma la matriz de características para incluir solo las características seleccionadas.
        
        Parámetros:
        - features: Matriz de características original.
        
        Retorna:
        - Matriz de características seleccionadas.
        """
        return features[self.selected_features]
    
    def fit_transform(self, target, features, only_list_features = False):
        """
        >>> METODO PRINCIPAL.
        
        Ajusta el modelo y transforma la matriz de características.
        
        Parámetros:
        - target: Etiquetas objetivo.
        - features: Matriz de características.
        
        Retorna:
        - Si labels_return es False: Matriz de características seleccionadas.
        - Si labels_return es True: Tupla (matriz de características seleccionadas, etiquetas).
        """
        # Executing Feature Importance
        self.fit(target, features)
        # If user wants only the list of features
        if only_list_features:
            # Return list of features
            return self.selected_features
        else: 
            # Selecciona las columnas con los features ideales
            features_selected = self.transform(features)
            
            # Si el usuario desea retornar tambien los labels
            if self.labels_return:
                return (features_selected, target)
            # De lo contrario, solo el df con features seleccionados
            else:
                return features_selected