@echo off
echo ===================================================
echo Configurando entorno para el Proyecto MLOps
echo ===================================================

echo 1. Creando entorno virtual...
python -m venv venv

echo 2. Activando entorno virtual...
call venv\Scripts\activate

echo 3. Instalando dependencias desde requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt

echo ===================================================
echo ¡Configuración completada con exito!
echo Para activar el entorno en el futuro usa: venv\Scripts\activate
echo ===================================================
pause
