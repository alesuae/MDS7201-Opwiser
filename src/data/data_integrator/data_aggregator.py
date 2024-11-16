import pandas as pd
from utils.config import get_config
import yaml

# TODO: remove hardcoded data!!
class DataAggregator:
    def __init__(self, config_mode):
        self.config_mode = config_mode
        self.config_dict = get_config(config_mode)
        self.config_path = self.config_dict['config_path']
        # Inicialización de niveles de agregación
        self.aggregation_level = self.config_dict['aggregation_level']
        self.aggregation_methods = self.config_dict['data']['aggregation_methods']

    def interactive_aggregation_setup(self, df):
        """
        Configura métodos de agregación de forma interactiva si no están definidos en el YAML.
        """
        if self.aggregation_methods:
            # Si ya existen métodos definidos, usarlos directamente
            print("\nMétodos de agregación cargados desde el YAML:")
            for var, method in self.aggregation_methods.items():
                print(f" - {var}: {method}")
            use_existing = input("\n¿Usar estos métodos? (s/n): ").strip().lower()
            if use_existing == 's':
                return
        else:
            print("\nNo hay métodos de agregación definidos. Configurando de forma interactiva...")

        print("\nVariables disponibles para agregación:")
        variables = df.columns.tolist()
        aggregation_methods = {}

        for i, var in enumerate(variables):
            print(f"{i + 1}. {var}")

        print("\nConfigura las variables para agregar:")
        for var in variables:
            include = input(f"Incluir '{var}' en la agregación? (s/n): ").strip().lower()
            if include == 's':
                method = input(f"Método de agregación para '{var}' (e.g., sum, mean, max, min): ").strip()
                aggregation_methods[var] = method

        # Confirmar configuración
        print("\nMétodos de agregación seleccionados:")
        for var, method in aggregation_methods.items():
            print(f" - {var}: {method}")

        # Guardar en el YAML
        self.save_to_yaml(aggregation_methods)
        self.aggregation_methods = aggregation_methods

    def save_to_yaml(self, aggregation_methods):
        """
        Guarda los métodos de agregación en el archivo `data.config.yaml`.
        """
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

        if self.config_mode not in config:
            config[self.config_mode] = {}

        config[self.config_mode]['aggregation_methods'] = aggregation_methods

        with open(self.config_path, 'w') as file:
            yaml.dump(config, file)

        print(f"\nMétodos de agregación guardados en {self.config_path}")

    def aggregate(self, df):
        """
        Realiza la agregación según la configuración.
        """
        # Configurar nivel de agregación
        if self.aggregation_level == 'monthly':
            df['mes'] = pd.to_datetime(df['fecha']).dt.to_period('M')
            groupby_cols = ['mes', 'codigo_producto2']
        elif self.aggregation_level == 'weekly':
            df['semana'] = pd.to_datetime(df['fecha']).dt.to_period('W')
            groupby_cols = ['semana', 'codigo_producto2']
        else:
            raise ValueError(f"Nivel de agregación '{self.aggregation_level}' no encontrado.")

        # Aplicar métodos de agregación configurados
        aggregated_df = df.groupby(groupby_cols).agg(self.aggregation_methods).reset_index()

        return aggregated_df
