#Imports conjunto
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, f_oneway
import seaborn as sns
import matplotlib.pyplot as plt

def describe_df(df_origen):
    '''La función recibe un dataframe origen y devuelve un dataframe resultado 
    con información sobre el tipo de dato, valores faltantes, valores únicos y cardinalidad
    
    Argumento: 
    1. Parámetro único: DataFrame a analizar

    Retorna:
    1. Nombre de la variable
    2. Tipo de dato de la variable
    3. Porcentaje de valores nulos de la variable
    4. Número de valores únicos de la variable
    5. Porcentaje de cardinalidad de la variable


    '''
    # Creamos el diccionario para almacenar los resultados de los indicadores:
    resultado = {
        "COL_N": [],
        "DATA_TYPE": [],
        "MISSINGS (%)": [],
        "UNIQUE_VALUES":[],
        "CARDIN (%)": []
    }
    # Rellenamos los valores iterando en las columnas del DataFrame de origen:
    for col in df_origen.columns:
        resultado["COL_N"].append(col)
        resultado["DATA_TYPE"].append(df_origen[col].dtype)
        missings = round(df_origen[col].isna().sum()/len(df_origen)*100, 1)
        resultado["MISSINGS (%)"].append(missings)
        valores_unicos=df_origen[col].nunique()
        resultado["UNIQUE_VALUES"].append(valores_unicos)
        cardinalidad = round((valores_unicos/len(df_origen))*(1-missings/100),2)
        resultado["CARDIN (%)"].append(cardinalidad)
    
    df_resultado = pd.DataFrame(resultado) # convertimos en un DataFrame

    df_resultado.set_index("COL_N", inplace=True) # Establecemos como indices los nombres de las variables


    return df_resultado.T #Trasponemos el DataFrame


def clasifica_variables(df, umbral_categoria, umbral_continua):
    """
    Clasifica las columnas de un DataFrame en Binaria, Categórica, Numérica Continua o Numérica Discreta.

    Argumentos:
    df (pd.DataFrame): El DataFrame a analizar.
    umbral_categoria (int): Umbral de cardinalidad para diferenciar entre categórica y numérica.
    umbral_continua (float): Umbral del porcentaje de cardinalidad para diferenciar entre numérica continua y discreta.

    Retorna:
    pd.DataFrame: DataFrame con las columnas 'nombre_variable' y 'tipo_sugerido'.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El argumento proporcionado no es un DataFrame.")

    total_filas = len(df)
    resultados = []

    for col in df.columns:
        cardinalidad = df[col].nunique(dropna=True)  # Ignorar valores nulos para el conteo único
        porcentaje_cardinalidad = cardinalidad / total_filas if total_filas > 0 else 0
        es_numerica = pd.api.types.is_numeric_dtype(df[col])

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad == 1:
            tipo = "Constante"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif cardinalidad >= umbral_categoria:
            if es_numerica:
                tipo = "Numérica Continua" if porcentaje_cardinalidad >= umbral_continua else "Numérica Discreta"
            else:
                tipo = "Categórica de excesiva cardinalidad"
        else:
            tipo = "Indefinida"

        resultados.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(resultados)




def get_features_num_regression(dataframe,target_col,umbral_corr: float,pvalue=None,umbral_cat=20):
    """
    Esta función Selecciona las columnas numéricas de un DataFrame cuya correlación con la columna Target
    sea superior al umbral especificado. De manera opcional, aplica un test de hipótesis para asegurar que las
    correlaciones son estadísticamente significativas.

    Argumentos:
    dataframe (pd.DataFrame): DataFrame de entrada que contiene las variables a analizar.
    target_col (str): Nombre de la columna objetivo (debe ser numérica continua).
    umbral_corr (float): Umbral mínimo de correlación en el rango [0,1].
    pvalue (float, opcional): Nivel de significancia estadística. Si es None, no se aplica el test de hipótesis.

    Retorna:
    list: Lista de columnas numéricas que cumplen con el criterio de correlación y, si se especifica,
          la significancia estadística.
    """
    #COMPROBACION DE QUE LOS VALORES INTRODUCIDOS CUMPLEN CON LOS REQUISITOS
    if not isinstance(umbral_cat, int):
        print("El argumento 'umbral_cat' debe ser de tipo int")
        return None
    card_targ=dataframe[target_col].nunique()
    if (dataframe[target_col].dtype not in [np.int64, np.float64]) or card_targ<umbral_cat:
        print("La columna target debe ser numerica continua. types validos: [int64,float64]")
        return None
    elif not isinstance(umbral_corr, float) or not (0 <= umbral_corr <= 1):
        print("El argumento 'umbral_corr' debe ser de tipo float y estar entre los valores [0,1]")
        return None
    elif pvalue is not None and (not isinstance(pvalue, float) or not (0 <= pvalue <= 1)):
        print("El argumento 'pvalue' debe ser None o de tipo float y estar entre los valores [0,1]")
        return None
    
    else:
        #ESTUDIO DE LA CORRELACION ENTRE LAS COLUMNAS NUMERICAS Y LA TARGET_COL.
        df_clasificacion=clasifica_variables(dataframe.drop(columns=target_col),umbral_cat,0.05)
        numericas= df_clasificacion[(df_clasificacion["tipo_sugerido"]=="Numérica Continua") | (df_clasificacion["tipo_sugerido"]=="Numérica Discreta")]["nombre_variable"].to_list()
        features_num=[]
        print(f"La correlacion entre las columnas numericas y el target debe superar: {umbral_corr}")
        print("---------------------------------------------------------------------------")

        for col in numericas:
            if dataframe[col].isnull().sum() > 0:
                print(f"Advertencia: La columna <{col}> contiene valores nulos, no será tenida en cuenta.")
                continue
            correlation_w_target=dataframe[col].corr(dataframe[target_col])
            print(f"<{col}> corr con target: {correlation_w_target}")
            if np.abs(correlation_w_target)>=umbral_corr:
                features_num.append(col)

        #ESTUDIO DE LA SIGNIFICANCIA ESTADISTICA DE LAS CORRELACIONES.
        features_num_filtrada = features_num[:]
        if pvalue is not None:
            features_num_filtrada=[]
            nivel_significancia = 1 - pvalue
            print("\n¿Es la correlacion estadisticamente significativa?")
            print("---------------------------------------------------------------------------")
            for col in features_num:
                corr, valor_p = pearsonr(dataframe[col], dataframe[target_col])
                if valor_p < nivel_significancia:
                    features_num_filtrada.append(col)
                    print(f"<{col}>: p_value = {valor_p}  Si")
                else:
                    print(f"<{col}>: No")

    return features_num_filtrada
            



def plot_features_num_regression(dataframe, target_col="", columns=[], umbral_corr=0, pvalue=None):
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("El argumento 'dataframe' debe ser un DataFrame de pandas.")
    if not isinstance(target_col, str) or not target_col:
        raise ValueError("El argumento 'target_col' debe ser un string no vacío.")
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise ValueError("El argumento 'columns' debe ser una lista de strings.")
    if not isinstance(umbral_corr, (int, float)):
        raise ValueError("El argumento 'umbral_corr' debe ser un número.")
    if pvalue is not None and not (isinstance(pvalue, (int, float)) and 0 <= pvalue <= 1):
        raise ValueError("El argumento 'pvalue' debe ser un número entre 0 y 1 o 'None'.")
    
    # Si columns está vacío, tomar las columnas numéricas
    df_clasificacion=clasifica_variables(dataframe,20,0.05)
    if not columns:
        columns = df_clasificacion[(df_clasificacion["tipo_sugerido"]=="Numérica Continua") | (df_clasificacion["tipo_sugerido"]=="Numérica Discreta")]["nombre_variable"].to_list()
        if target_col in columns:
            columns.remove(target_col)  # eliminar target_col si está en la lista de columnas
    
    valid_columns = []
    for col in columns:
        # Eliminar filas con NaN para target_col y col
        filtered_df = dataframe[[target_col, col]].dropna()
        
        if filtered_df.shape[0] > 1:  # Solo proceder si hay más de un dato
            corr_coef = filtered_df.corr().iloc[0, 1]
            if abs(corr_coef) > umbral_corr:
                if pvalue is not None:
                    try:
                        corr_test_pvalue = pearsonr(filtered_df[target_col], filtered_df[col])[1]
                        print(f"Columna: {col}, p-value: {corr_test_pvalue}")  # Añadido
                        # Validar si el p-value es escalar
                        if isinstance(corr_test_pvalue, (float, int)) and corr_test_pvalue <= (1 - pvalue):
                            valid_columns.append(col)
                    except Exception as e:
                        print(f"Error al calcular el p-value para la columna {col}: {e}")
                else:
                    valid_columns.append(col)
    
    # División de columnas en grupos de hasta cinco, incluyendo target_col en cada uno
    groups = [valid_columns[i:i+4] for i in range(0, len(valid_columns), 4)]
    for group in groups:
        plot_cols = [target_col] + group
        sns.pairplot(dataframe, vars=plot_cols, diag_kind='kde')
        plt.show()
    
    return valid_columns

def get_features_cat_regression(dataframe, target_col, pvalue=0.05):
    """
    Identifica las columnas categóricas en un DataFrame que tienen una relación significativa con una columna objetivo numérica, basada en un nivel de confianza estadístico.

    Argumentos:
    -Dataframe: El conjunto de datos que contiene las columnas a analizar.
    -Target: El nombre de la columna objetivo, que debe ser numérica.
    -pvalue: El nivel de significación estadística para considerar una relación significativa
                

    Retorna: Una lista con los nombres de las columnas categóricas que tienen una relación estadísticamente significativa con la columna objetivo.
    """
    #Verifica si target_col es numérica
    if (dataframe[target_col].dtype not in [np.int64, np.float64]):
        print(f"La columna '{target_col}' no es numérica.")
        return None

    #Verifica la cardinalidad de target_col
    if dataframe[target_col].nunique() < 20:
        print("La columna objetivo debe tener al menos 20 valores únicos.")
        return None

    #Verifica si pvalue está en el rango válido
    if not (0 < pvalue <= 1):
        print("El valor de 'pvalue' debe estar entre 0 y 1.")
        return None

    #Filtra las columnas categóricas
    df_clasificacion=clasifica_variables(dataframe,20,0.05)
    cat_columns = df_clasificacion[(df_clasificacion["tipo_sugerido"]=="Categórica") | (df_clasificacion["tipo_sugerido"]=="Binaria")]["nombre_variable"].to_list()

    #Aplica pruebas estadísticas para determinar la relación
    related_columns = []
    for col in cat_columns:
                
        try:
            dataframe[col] = dataframe[col].fillna("Desconocido")
            # Realiza ANOVA para evaluar la relación entre la categórica y la numérica
            groups = [dataframe[dataframe[col] == category][target_col] for category in dataframe[col].unique()]
            stat, p = f_oneway(*groups)

            # Agrega columna si el p-valor es menor al nivel de significación
            if p < pvalue:
                related_columns.append(col)
        except Exception as e:
            print(f"No se pudo evaluar la columna '{col}': {e}")

    return related_columns

def plot_features_cat_regression(dataframe, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    '''La función recibe un dataframe y analiza las variables categoricas significativas con la variable target, 
    si no detecta variables categoricas significativas, ejecuta analisis de variables categoricas significativas 
    con target mostrando histograma de los datos
    
    Argumentos: 
    1. dataframe: DataFrame a analizar
    2. target_col: variable objetivo de estudio
    3. columns: por defecto vacia, son las variables categoricas a analizar. 
    4. pvalue: pvalue que por defecto se establece en 0.05
    5. with_indivual_plot : indica si queremos generar y mostrar un histograma separado 
    por cada variable categorica significativa, por defecto False: se presentan agrupadas

    Retorna:
    1. Si no hay variables categóricas, ejecuta la función get_features_num_regresion
    2. Si hay variables categóricas, pintamos los histogramas de la variable target con cada una de las variables categóricas significativas
        2.1 individuales si hemos marcado with_individual_plot = True
        2.2 por defecto de forma agrupada'''
    
    # Establecemos la lista de variables categóricas significativas:
    columns_cat_significativas = []
    # En la función get_features_cat_regression hemos definido las variables categóricas significativas, 
    # la llamamos para comprobar si nuestras variables están en la lista de variables categóricas significativas.
    columnas_cat = get_features_cat_regression(dataframe, target_col, pvalue=pvalue)
    # Validamos si cumplen con el criterio de significación cada variable, se incorporan solo las que cumplen.
    for col in columns:
        if col in columnas_cat:
            columns_cat_significativas.append(col)

    # Si no hay ningún elemento en la lista:
    if len(columns_cat_significativas) == 0:
        print("No hay variables categóricas significativas")
        return []
    # Si tenemos variables categóricas significativas a analizar, pintamos los histogramas, 
    # agrupados o por variable categórica frente al target
    # Plotting de las variables categóricas significativas
    if with_individual_plot:
        for col in columns_cat_significativas:
            plt.figure(figsize=(12, 8))
            sns.histplot(data=dataframe, x=target_col, hue=col, multiple="dodge", 
                         palette="viridis", alpha=0.6, kde=True, fill= True)
            plt.title(f'Histograma de {target_col} por {col}', fontsize=16)
            plt.xlabel(target_col, fontsize=14)
            plt.ylabel('Frecuencia', fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title=col, labels=dataframe[col].unique())
            plt.show()
    else:
        # Crear subplots para cada variable categórica significativa en un solo cuadro
        num_plots = len(columns_cat_significativas)
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, 8 * num_plots))
        
        # Convertir axs a una lista si num_plots es 1
        if num_plots == 1:
            axs = [axs]
        
        for i, col in enumerate(columns_cat_significativas):
            sns.histplot(data=dataframe, x=target_col, hue=col, multiple="dodge", 
                         palette="viridis", alpha=0.6, kde=True, ax=axs[i], fill = True)
            axs[i].set_title(f'Histograma de {target_col} por {col}', fontsize=16)
            axs[i].set_xlabel(target_col, fontsize=14)
            axs[i].set_ylabel('Frecuencia', fontsize=14)
            axs[i].legend(title=col,labels=dataframe[col].unique())
            axs[i].tick_params(axis='x', rotation=45)
              
        plt.show()
                    
    return  columns_cat_significativas




# ========================================FUNCIONES DE REPRESENTACIONES GRAFICAS======================================================= #



def pinta_distribucion_categoricas(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia Relativa')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_categorical_relationship_fin(df, cat_col1, cat_col2, relative_freq=False, show_values=False, size_group = 5):
    # Prepara los datos
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    # Convierte a frecuencias relativas si se solicita
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    # Si hay más de size_group categorías en cat_col1, las divide en grupos de size_group
    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > size_group:
        num_plots = int(np.ceil(len(unique_categories) / size_group))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * size_group:(i + 1) * size_group]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=data_subset, order=categories_subset)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {cat_col1} y {cat_col2} - Grupo {i + 1}')
            plt.xlabel(cat_col1)
            plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de size_group categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=count_data)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {cat_col1} y {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


def plot_categorical_numerical_relationship(df, categorical_col, numerical_col, show_values=False, measure='mean'):
    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    # Si hay más de 5 categorías, las divide en grupos de 5
    if grouped_data.shape[0] > 5:
        unique_categories = grouped_data.index.unique()
        num_plots = int(np.ceil(len(unique_categories) / 5))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * 5:(i + 1) * 5]
            data_subset = grouped_data.loc[categories_subset]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=data_subset.index, y=data_subset.values)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
            plt.xlabel(categorical_col)
            plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de 5 categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=grouped_data.index, y=grouped_data.values)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {categorical_col} y {numerical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None):
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

def plot_grouped_boxplots(df, cat_col, num_col):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    group_size = 5

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=num_col, data=subset_df)
        plt.title(f'Boxplots of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xticks(rotation=45)
        plt.show()



def plot_grouped_histograms(df, cat_col, num_col, group_size):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        for cat in subset_cats:
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], kde=True, label=str(cat))
        
        plt.title(f'Histograms of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()



def grafico_dispersion_con_correlacion(df, columna_x, columna_y, tamano_puntos=50, mostrar_correlacion=False):
    """
    Crea un diagrama de dispersión entre dos columnas y opcionalmente muestra la correlación.

    Args:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    columna_x (str): Nombre de la columna para el eje X.
    columna_y (str): Nombre de la columna para el eje Y.
    tamano_puntos (int, opcional): Tamaño de los puntos en el gráfico. Por defecto es 50.
    mostrar_correlacion (bool, opcional): Si es True, muestra la correlación en el gráfico. Por defecto es False.
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=columna_x, y=columna_y, s=tamano_puntos)

    if mostrar_correlacion:
        correlacion = df[[columna_x, columna_y]].corr().iloc[0, 1]
        plt.title(f'Diagrama de Dispersión con Correlación: {correlacion:.2f}')
    else:
        plt.title('Diagrama de Dispersión')

    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.grid(True)
    plt.show()


def bubble_plot(df, col_x, col_y, col_size, scale = 1000):
    """
    Crea un scatter plot usando dos columnas para los ejes X e Y,
    y una tercera columna para determinar el tamaño de los puntos.

    Args:
    df (pd.DataFrame): DataFrame de pandas.
    col_x (str): Nombre de la columna para el eje X.
    col_y (str): Nombre de la columna para el eje Y.
    col_size (str): Nombre de la columna para determinar el tamaño de los puntos.
    """

    # Asegúrate de que los valores de tamaño sean positivos
    sizes = (df[col_size] - df[col_size].min() + 1)/scale

    plt.scatter(df[col_x], df[col_y], s=sizes)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f'Burbujas de {col_x} vs {col_y} con Tamaño basado en {col_size}')
    plt.show()


