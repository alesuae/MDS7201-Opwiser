class ExogenousDataSelector:
    def __init__(self, exogenous_data):
        """
        Inicializa el selector y muestra las columnas disponibles.
        """

        # Mostrar columnas disponibles
        self.variables = exogenous_data.columns.tolist()
        print("\nVariables disponibles en el dataset exógeno:")
        for i, var in enumerate(self.variables):
            print(f"{i + 1}. {var}")

        # Inicializa las variables seleccionadas desde el YAML (si existen)
        self.selected_variables = None

    def select(self, exogenous_data):
        """
        Filtra el dataset para incluir solo las variables seleccionadas.
        """
        if not self.selected_variables:
            print("\nSelecciona las variables que deseas incluir:")
            print(" - Ingresa los índices separados por comas (e.g., 1,2,3)")
            print(" - Presiona Enter para seleccionar TODAS las variables")
            
            # Leer entrada del usuario
            user_input = input("Índices (Enter para todas): ").strip()
            
            # Si el usuario presiona Enter, selecciona todas las variables
            if not user_input:
                self.selected_variables = self.variables
            else:
                # Seleccionar variables basadas en índices ingresados
                selected_indices = [int(i.strip()) - 1 for i in user_input.split(",")]
                self.selected_variables = [self.variables[i] for i in selected_indices]

            # Confirmación
            print(f"\nVariables seleccionadas: {self.selected_variables}")

        # Siempre incluir la columna de fecha si existe
        if "fecha" in exogenous_data.columns and "fecha" not in self.selected_variables:
            self.selected_variables.append("fecha")

        # Filtrar dataset
        return exogenous_data[self.selected_variables]
