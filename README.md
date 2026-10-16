# California_Housing
#Despliegue del modelo de precios de casas en California en Render

## 1. Preparación del repositorio
1. Crea un repositorio público en GitHub para alojar el proyecto.
2. Sube el archivo `app.py` con el código de la API desarrollado en FastAPI.
3. Comprueba que `modelo_california.pkl` está en la raíz del repositorio o disponible mediante una URL de GitHub.
4. Configura `MODEL_PATH = "modelo_california.pkl"` y establece `MODEL_URL` con la URL directa del modelo.
5. Crea un archivo `requirements.txt` con las dependencias: `fastapi`, `uvicorn`, `pandas`, `requests`, `joblib` y `scikit-learn`.
6. Guarda y sube todos los archivos necesarios al repositorio.

## 2. Configuración del servicio en Render
7. Accede a [Render](https://render.com/) y crea una cuenta o inicia sesión.
8. Selecciona **New** y, a continuación, **Web Service**.
9. Conecta tu cuenta de GitHub y selecciona el repositorio del proyecto.
10. Asigna un nombre al servicio y selecciona Python como entorno de ejecución.
11. Configura el comando de compilación como `pip install -r requirements.txt`.
12. Configura el comando de arranque como `uvicorn app:app --host 0.0.0.0 --port $PORT`.
13. Selecciona el plan de alojamiento deseado y crea el servicio.
14. Espera a que Render instale las dependencias y complete el despliegue.

## 3. Verificación del despliegue
15. Consulta los registros de Render para comprobar que la aplicación se inicia sin errores.
16. Verifica que el modelo se descarga desde GitHub si no existe en el entorno local.
17. Accede a la URL pública asignada por Render y comprueba que `/` devuelve `{"status": "ok"}`.
18. Abre `/docs` para acceder a la documentación interactiva de FastAPI.
19. Utiliza `POST /predict` para enviar las ocho variables de entrada y obtener una predicción.
20. Si el despliegue falla, revisa los registros, la URL del modelo y las dependencias instaladas.
