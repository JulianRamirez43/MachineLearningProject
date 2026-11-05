#!/bin/bash
echo "==================================================="
echo "Configurando entorno para el Proyecto MLOps (Mac/Linux)"
echo "==================================================="

echo "1. Creando entorno virtual..."
python3 -m venv venv

echo "2. Activando entorno virtual..."
source venv/bin/activate

echo "3. Instalando dependencias desde requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==================================================="
echo "¡Configuración completada con éxito!"
echo "Para activar el entorno en el futuro usa: source venv/bin/activate"
echo "==================================================="