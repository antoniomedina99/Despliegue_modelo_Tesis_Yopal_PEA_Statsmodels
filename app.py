import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Path del modelo preentrenado
MODEL_PATH = 'models/model_OLS'

# Función para recibir los datos de entrada y devolver la predicción
def model_prediction(x_in, model):
    # Convertir los datos de entrada a un DataFrame
    columns = ['const', 'q', 'f', 'x1', 'x2', 'g', 'w', 'z']
    x_df = pd.DataFrame([x_in], columns=columns)

    # Realizar la predicción
    preds = model.predict(x_df)

    # Convertir el logaritmo del ingreso a ingreso real
    preds = np.exp(preds)

    return preds

def main():
    
    model = None

    # Cargar el modelo
    if model is None:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    
    # Título
    st.markdown("""
    <h1 style="color:#181082;text-align:center;">ESTIMACIÓN DEL INGRESO PROMEDIO DE LA POBLACIÓN ECONÓMICAMENTE ACTIVA OCUPADA EN YOPAL CASANARE</h1>
    """, unsafe_allow_html=True)

    # Descripción de donde encontrar el código 
    st.markdown("""
    Completa el cuestionario para obtener un una proyección de tus ingreso. En la última sección, encontrarás un enlace al código del proyecto en GitHub.
    """, unsafe_allow_html=True)



    # Interpretación de resultados
    st.markdown("""
    <h2 style="color:#181082;text-align:center;">Interpretación de Resultados</h2>
    <p style="text-align:left;">
        El modelo fue estimado mediante errores estándar robustos, el cual presenta la siguiente ecuación:
    </p>
    """, unsafe_allow_html=True)

    # Ecuación del modelo econométrico
    st.latex(r'''
    \text{Modelo Econométrico:} \\
    \hat\ln(Y) = \hat{\beta}_0 + \hat{\beta}_1Q + \hat{\beta}_2F + \hat{\beta}_3X_1 + \hat{\beta}_4X_2 + \hat{\beta}_5G + \hat{\beta}_6W + \hat{\beta}_7Z + \epsilon_t
    ''')

    # Ecuación del modelo estimado
    st.latex(r'''
    \text{Modelo Estimado:} \\
    {YLn} = {\beta}_0 + {\beta}_1Q + {\beta}_2F + {\beta}_3X_1 + {\beta}_4X_2 + {\beta}_5G + {\beta}_6W + {\beta}_7Z
    ''')

    st.markdown("""
    <h2 style="color:#181082;text-align:left;">Simulación</h2>
    <p style="text-align:left;">
        La ecuación resultante del modelo calculando con errores robustos presenta los siguientes estimados:
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""Función de regresión múltiple:""")
    st.latex(r'''
    \hat{YLn}= {\beta_0}+ {\beta}_1Q+ {\beta}_2F+ {\beta}_3X_1+ {\beta}_4X_2+ {\beta}_5G_1+ t{\beta_6}W+ {\beta_7}
    ''')

    st.markdown("""Donde:""")

    st.latex(r'''
    \hat{YLn}= 14.106158-0.125724*Q+ 0.2490846*F -0.5176395*X_1 - \\
            0.3715330*X_2 -  0.1556549*G + 0.0091678*W + 0.0080724*Z
             
            ''')
    st.markdown("""Así mismo:""")
    
    st.latex(r'''
    \hat{YLn}= 14.1061586 - 0.1257248*Q- 0.5176395*X_1 - 0.3715330*X_2+ \\
            0.2490846*F + 0.1556549*G + 0.0091678*W + 0.0080724*Z 
    
             ''')

    st.latex(r'''
    \hat{YLn} = b
             ''')   

    st.latex(r'''
    e^(\hat{YLn})= e^b
             ''')
    
    st.latex(r'''
    \hat{Y} = c
             ''')           

    st.markdown("""
    <p style="text-align:left;">
        Por favor, marque la casilla correspondiente para realizar la estimación.
    </p>
    """, unsafe_allow_html=True)

    # Lectura de datos
    canal_buscado = st.radio("1.¿Consiguió su trabajo actual con recomendación?:", options=["Con recomendación", "Sin recomendación"])
    f = st.radio("2.¿Estudió fuera del departamento?:", options=["No", "Sí"])
    
    # Selección del nivel educativo
    nivel_educativo = st.selectbox("3.Nivel de escolaridad:", options=["Bachiller", "Técnico", "Profesional"])
    
    g = st.radio("3.Género:", options=["Masculino","Femenino"])
    w = st.number_input("4.Edad:", min_value=0)
    z = st.number_input("5.Experiencia laboral (en años):", min_value=0)
    
    # Convertir opciones a valores numéricos
    q = 1 if canal_buscado == "Con recomendación" else 0
    f_val = 1 if f == "Sí" else 0
    
    # Convertir nivel educativo a variables x1 y x2
    x1_val = 1 if nivel_educativo == "Bachiller" else 0
    x2_val = 1 if nivel_educativo == "Técnico" else 0
    # El valor de x3 (Profesional) se representa implícitamente si x1 y x2 son 0
    # No se necesita definir explícitamente x3 en el modelo

    g_val = 1 if g == "Masculino" else 0

    # Botón de predicción
    if st.button("Predicción :"): 
        # Crear la entrada del modelo
        x_in = [1,  # Constante
                q,      # Canal de búsqueda
                f_val,  # Estudió fuera del departamento
                x1_val, # Bachiller
                x2_val, # Técnico
                g_val,  # Género
                w,      # Edad
                z]      # Experiencia laboral
        
        predictS = model_prediction(x_in, model)
        st.success(f'EL INGRESO PROMEDIO ESTIMADO ES: ${int(predictS[0]):,} pesos colombianos')



    # Enlace al repositorio de GitHub donde se muestra el Notebook de la creación del modelo 

    st.markdown("""
    <h2 style="color:#181082;text-align:center;">Modelo econometrico</h2>
    <p style="text-align:center;">
        Puedes encontrar el análisis econométrico en el siguiente enlace: :
    </p>
                
    <p style="text-align:center;">
        <a href="https://github.com/antoniomedina99/Tesis_Yopal_PEA_Statsmodels/blob/main/notebooks/Reporte.ipynb" target="_blank">Repositorio de GitHub</a>
    </p>
    """, unsafe_allow_html=True)


    # Nota sobre el valor de la estimación 
    st.markdown("""
    *Nota: Los datos del estudio son estaticos y fueron recolectados en el año 2021, por lo tanto, la fidelidad de la estimación a medida que pasa el tiempo se hace menos exacta.*
    """, unsafe_allow_html=True)

    
if __name__ == '__main__':
    main()