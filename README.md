# Evaluación 2 - Fundamentos de Data Science
Profesor: Héctor Eduardo Cifuentes Mella
Sección: 302
Integrantes:
- Fancisco Esteban Solís Maturana
- Catalina Nomeí Beas Pérez
- María José Garía Huarca

# Instrucciones de Instalación del repositorio

<details>
<summary>Requisitos Técnicos</summary>

- [Visual Studio Code](https://code.visualstudio.com) - [Descarga](https://code.visualstudio.com/Download)
- [Python](https://python.org) - [Descarga](https://www.python.org/downloads/)
- [Git](https://git-scm.com) - [Descarga](https://git-scm.com/install/)
</details>

> Para los siguientes pasos si estás en Windows utiliza la app `Git Bash`, de otra forma solo usa tu Terminal.

### 1. Clonar el repositorio
Utiliza este comando para clonar el repositorio:
```bash
git clone https://github.com/Im-Fran/evaluacion-2-fundamentos-de-data-science.git
```
Esto creará una carpeta `evaluacion-2-fundamentos-de-data-science` donde tendrás el código para trabajar.

Luego para asegurarte que estás dentro de la carpeta del repositiorio para trabajar usa `cd evaluacion-2-fundamentos-de-data-science`

###  2. Crea el perfil vitual de python.
Ya que es muy probable que tengas otros proyectos dentro de tu computador, para evitar incompatibilidades se utilizan los entornos virtuales. En nuestro caso usaremos `venv`.
Para inicializar tu entorno usa `python3 -m venv .venv`.

Esto generará una carpeta `.venv/` la cual contendrá varios archivos. Deberás de ejecutar este comando para entrar en el entorno:
```bash
source .venv/bin/activate
```
Una vez dentro si ejecutas este comando: `which python` deberías tener una salida similar a esta:
```log
.../evaluacion-2-fundamentos-de-data-science/.venv/bin/python
```

Si tienes este ouput, genial! Eso significa que estas usando el entorno virtual. Antes de comenzar a trabajar recuerda siempre ejecutar el comando `source .venv/bin/activate`.
> Si utilizas un IDE o un editor como Visual Studio Code, generalmente (al tener las extensiones de python) se activa automáticamente el entorno virtual.


### 3. Instala las dependencias
Para instalar los paquetes requeridos usa este comando:
```bash
pip install -r requirements.txt
```

Si ves una advertencia como esta, puedes ignorarla:
```log
❯ pip install -r requirements.txt

[notice] A new release of pip is available: 24.3.1 -> 25.3
[notice] To update, run: pip install --upgrade pip
```

Esto significa que hay una nueva version de pip (el administrador de paquetes de python). Si gustas, puedes actualizarla usando el comando indicado `pip install --upgrade pip`
