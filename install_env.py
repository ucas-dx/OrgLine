import os
import subprocess
import yaml
import sys

def install_conda_package(package):
    try:
        subprocess.run(['conda', 'install', '-y', package], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing Conda package: {package}\n{e}")

def install_pip_package(package):
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing pip package: {package}\n{e}")

def install_packages_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        env_data = yaml.safe_load(file)

    # Install Conda packages
    for dep in env_data['dependencies']:
        if isinstance(dep, str):
            install_conda_package(dep)
        elif isinstance(dep, dict) and 'pip' in dep:
            pip_packages = dep['pip']
            for pip_package in pip_packages:
                install_pip_package(pip_package)

def main():
    yaml_file = 'environment.yaml'
    if not os.path.exists(yaml_file):
        print(f"YAML file '{yaml_file}' not found.")
        return

    install_packages_from_yaml(yaml_file)

if __name__ == '__main__':
    main()
